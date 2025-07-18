#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 이미지 처리 유틸리티 모듈

이 모듈은 PDF 파일을 이미지로 변환하고, 이미지 품질을 개선하며,
OCR 처리에 최적화된 이미지 전처리 기능을 제공합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import math
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import numpy as np
from dataclasses import dataclass
from enum import Enum

from core.base_classes import BaseProcessor
from core.exceptions import (
    ImageProcessingError,
    FileProcessingError,
    ProcessingError,
    ValidationError,
    ApplicationError,
)
from config.constants import (
    PDF_FILE_EXTENSIONS,
    IMAGE_FILE_EXTENSIONS,
    DEFAULT_IMAGE_RESOLUTION,
    IMAGE_DEFAULT_QUALITY,
    IMAGE_MAX_DIMENSION,
    IMAGE_MIN_DIMENSION,
    IMAGE_NOISE_REDUCTION_KERNEL_SIZE,
    IMAGE_BLUR_KERNEL_SIZE,
    IMAGE_SHARPENING_STRENGTH,
    FILE_PROCESSING_TIMEOUT_SECONDS,
)
from config.settings import ApplicationConfig
from utils.logger_util import get_application_logger


class ImageFormat(Enum):
    """지원되는 이미지 형식 열거형"""

    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"


class ImageColorSpace(Enum):
    """이미지 색상 공간 열거형"""

    RGB = "rgb"
    RGBA = "rgba"
    GRAYSCALE = "grayscale"
    CMYK = "cmyk"


class ImageQuality(Enum):
    """이미지 품질 등급 열거형"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


@dataclass
class ImageProcessingOptions:
    """이미지 처리 옵션 클래스"""

    # 기본 설정
    output_format: ImageFormat = ImageFormat.PNG
    output_quality: int = IMAGE_DEFAULT_QUALITY
    target_resolution: int = DEFAULT_IMAGE_RESOLUTION
    color_space: ImageColorSpace = ImageColorSpace.RGB

    # 크기 조정 옵션
    target_width: Optional[int] = None
    target_height: Optional[int] = None
    maintain_aspect_ratio: bool = True

    # 품질 개선 옵션
    enable_noise_reduction: bool = True
    enable_sharpening: bool = True
    enable_contrast_enhancement: bool = True
    enable_brightness_adjustment: bool = True

    # 기하학적 보정 옵션
    enable_rotation_correction: bool = True
    enable_skew_correction: bool = True
    rotation_threshold: float = 0.5
    skew_threshold: float = 0.5

    # 출력 옵션
    create_backup: bool = True
    preserve_metadata: bool = True


# ====================================================================================
# 1. 메인 이미지 처리 클래스
# ====================================================================================


class ImageProcessor(BaseProcessor):
    """
    이미지 처리 클래스

    PDF 파일을 이미지로 변환하고, 이미지 품질을 개선하며,
    OCR 처리에 최적화된 전처리를 수행합니다.
    BaseProcessor를 상속받아 표준 처리 인터페이스를 구현합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        ImageProcessor 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)

        # 하위 처리기 초기화
        self.converter = ImageConverter(config, logger)
        self.enhancer = ImageEnhancer(config, logger)
        self.validator = ImageValidator(config, logger)

        # 처리 옵션 설정
        self.processing_options = ImageProcessingOptions()

        # 지원되는 형식 설정
        self.supported_pdf_formats = PDF_FILE_EXTENSIONS
        self.supported_image_formats = IMAGE_FILE_EXTENSIONS

        # 처리 통계
        self.images_processed = 0
        self.images_converted = 0
        self.images_enhanced = 0
        self.processing_errors = []

        self.logger.info("ImageProcessor initialized successfully")

    def process(self, data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        이미지 처리 메인 메서드

        Args:
            data: 처리할 파일 경로 또는 처리 옵션이 포함된 딕셔너리

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 입력 데이터 파싱
            if isinstance(data, str):
                file_path = data
                options = self.processing_options
            elif isinstance(data, dict):
                file_path = data.get("file_path")
                options = self._parse_processing_options(data)
            else:
                raise ValidationError("Invalid input data format")

            if not self.validate_input(file_path):
                raise ValidationError(f"Invalid input file: {file_path}")

            # 파일 형식에 따른 처리
            file_extension = Path(file_path).suffix.lower()

            if file_extension in self.supported_pdf_formats:
                return self._process_pdf_file(file_path, options)
            elif file_extension in self.supported_image_formats:
                return self._process_image_file(file_path, options)
            else:
                raise ImageProcessingError(
                    message=f"Unsupported file format: {file_extension}",
                    image_path=file_path,
                    processing_operation="format_check",
                )

        except Exception as e:
            self.processing_errors.append(str(e))
            self.logger.error(f"Image processing failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Image processing failed: {str(e)}",
                image_path=file_path if "file_path" in locals() else "unknown",
                processing_operation="process",
                original_exception=e,
            )

    def validate_input(self, data: Any) -> bool:
        """
        입력 데이터 검증

        Args:
            data: 검증할 데이터

        Returns:
            bool: 검증 결과
        """
        try:
            if not isinstance(data, str):
                return False

            if not os.path.exists(data):
                return False

            file_extension = Path(data).suffix.lower()
            supported_formats = (
                self.supported_pdf_formats + self.supported_image_formats
            )

            if file_extension not in supported_formats:
                return False

            # 파일 크기 확인
            file_size = os.path.getsize(data)
            if file_size == 0:
                return False

            return True

        except Exception as e:
            self.logger.error(f"Input validation failed: {str(e)}")
            return False

    def _process_pdf_file(
        self, pdf_path: str, options: ImageProcessingOptions
    ) -> Dict[str, Any]:
        """
        PDF 파일 처리

        Args:
            pdf_path: PDF 파일 경로
            options: 처리 옵션

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # PDF를 이미지로 변환
            converted_images = self.converter.convert_pdf_to_images(pdf_path, options)

            # 각 이미지 개선
            enhanced_images = []
            for image_path in converted_images:
                if options.enable_noise_reduction or options.enable_sharpening:
                    enhanced_path = self.enhancer.enhance_image_quality(
                        image_path, options
                    )
                    enhanced_images.append(enhanced_path)
                else:
                    enhanced_images.append(image_path)

            # 처리 결과 반환
            result = {
                "source_file": pdf_path,
                "file_type": "pdf",
                "converted_images": converted_images,
                "enhanced_images": enhanced_images,
                "total_pages": len(converted_images),
                "processing_options": options.__dict__,
                "processing_time": 0.0,  # 실제 처리 시간 계산 필요
            }

            self.images_converted += len(converted_images)
            self.images_enhanced += len(enhanced_images)

            return result

        except Exception as e:
            self.logger.error(f"PDF processing failed: {str(e)}")
            raise ImageProcessingError(
                message=f"PDF processing failed: {str(e)}",
                image_path=pdf_path,
                processing_operation="process_pdf",
                original_exception=e,
            )

    def _process_image_file(
        self, image_path: str, options: ImageProcessingOptions
    ) -> Dict[str, Any]:
        """
        이미지 파일 처리

        Args:
            image_path: 이미지 파일 경로
            options: 처리 옵션

        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            # 이미지 검증
            validation_result = self.validator.validate_image_format(image_path)
            if not validation_result:
                raise ImageProcessingError(
                    message=f"Image validation failed: {image_path}",
                    image_path=image_path,
                    processing_operation="validate_image",
                )

            # 이미지 개선
            enhanced_path = image_path
            if options.enable_noise_reduction or options.enable_sharpening:
                enhanced_path = self.enhancer.enhance_image_quality(image_path, options)

            # 기하학적 보정
            if options.enable_rotation_correction or options.enable_skew_correction:
                corrected_path = self.enhancer.correct_image_geometry(
                    enhanced_path, options
                )
                enhanced_path = corrected_path

            # 처리 결과 반환
            result = {
                "source_file": image_path,
                "file_type": "image",
                "enhanced_image": enhanced_path,
                "validation_result": validation_result,
                "processing_options": options.__dict__,
                "processing_time": 0.0,
            }

            self.images_processed += 1
            self.images_enhanced += 1

            return result

        except Exception as e:
            self.logger.error(f"Image processing failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Image processing failed: {str(e)}",
                image_path=image_path,
                processing_operation="process_image",
                original_exception=e,
            )

    def _parse_processing_options(self, data: Dict[str, Any]) -> ImageProcessingOptions:
        """
        처리 옵션 파싱

        Args:
            data: 옵션 데이터

        Returns:
            ImageProcessingOptions: 파싱된 옵션
        """
        options = ImageProcessingOptions()

        # 기본 설정 파싱
        if "output_format" in data:
            options.output_format = ImageFormat(data["output_format"])
        if "output_quality" in data:
            options.output_quality = data["output_quality"]
        if "target_resolution" in data:
            options.target_resolution = data["target_resolution"]

        # 크기 조정 옵션 파싱
        if "target_width" in data:
            options.target_width = data["target_width"]
        if "target_height" in data:
            options.target_height = data["target_height"]

        # 품질 개선 옵션 파싱
        if "enable_noise_reduction" in data:
            options.enable_noise_reduction = data["enable_noise_reduction"]
        if "enable_sharpening" in data:
            options.enable_sharpening = data["enable_sharpening"]

        return options

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        처리 통계 반환

        Returns:
            Dict[str, Any]: 처리 통계
        """
        return {
            "processor_id": self.processor_id,
            "processor_type": self.__class__.__name__,
            "images_processed": self.images_processed,
            "images_converted": self.images_converted,
            "images_enhanced": self.images_enhanced,
            "processing_errors": len(self.processing_errors),
            "error_messages": self.processing_errors[-10:],  # 최근 10개 오류
            "success_rate": (
                (self.images_processed - len(self.processing_errors))
                / max(self.images_processed, 1)
                * 100
            ),
        }


# ====================================================================================
# 2. 이미지 변환 클래스
# ====================================================================================


class ImageConverter:
    """
    이미지 변환 클래스

    PDF 파일을 이미지로 변환하고, 이미지 형식 변환을 처리합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        ImageConverter 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        self.config = config
        self.logger = logger
        self.temp_directory = tempfile.gettempdir()

    def convert_pdf_to_images(
        self,
        pdf_path: str,
        options: ImageProcessingOptions,
        output_dir: Optional[str] = None,
    ) -> List[str]:
        """
        PDF를 이미지로 변환

        Args:
            pdf_path: PDF 파일 경로
            options: 변환 옵션
            output_dir: 출력 디렉터리 (None이면 임시 디렉터리 사용)

        Returns:
            List[str]: 변환된 이미지 파일 경로 목록
        """
        try:
            if not os.path.exists(pdf_path):
                raise FileProcessingError(
                    message=f"PDF file not found: {pdf_path}",
                    file_path=pdf_path,
                    operation="convert_pdf_to_images",
                )

            # 출력 디렉터리 설정
            if output_dir is None:
                output_dir = self.temp_directory

            os.makedirs(output_dir, exist_ok=True)

            # PDF 변환 수행
            converted_images = self._convert_with_library(pdf_path, options, output_dir)

            self.logger.info(
                f"PDF converted to {len(converted_images)} images: {pdf_path}"
            )
            return converted_images

        except Exception as e:
            self.logger.error(f"PDF to image conversion failed: {str(e)}")
            raise ImageProcessingError(
                message=f"PDF to image conversion failed: {str(e)}",
                image_path=pdf_path,
                processing_operation="convert_pdf_to_images",
                original_exception=e,
            )

    def _convert_with_library(
        self, pdf_path: str, options: ImageProcessingOptions, output_dir: str
    ) -> List[str]:
        """
        라이브러리를 사용한 PDF 변환

        Args:
            pdf_path: PDF 파일 경로
            options: 변환 옵션
            output_dir: 출력 디렉터리

        Returns:
            List[str]: 변환된 이미지 파일 경로 목록
        """
        converted_images = []

        try:
            # PyMuPDF 사용 시도
            try:
                import fitz  # PyMuPDF

                pdf_document = fitz.open(pdf_path)

                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]

                    # 변환 매트릭스 설정
                    mat = fitz.Matrix(
                        options.target_resolution / 72.0,  # X 스케일
                        options.target_resolution / 72.0,  # Y 스케일
                    )

                    # 페이지를 이미지로 변환
                    pix = page.get_pixmap(matrix=mat)

                    # 이미지 저장
                    output_filename = f"{Path(pdf_path).stem}_page_{page_num + 1:03d}.{options.output_format.value}"
                    output_path = os.path.join(output_dir, output_filename)

                    pix.save(output_path)
                    converted_images.append(output_path)

                    self.logger.debug(f"Page {page_num + 1} converted: {output_path}")

                pdf_document.close()
                return converted_images

            except ImportError:
                self.logger.warning("PyMuPDF not available, trying alternative method")
                return self._convert_with_alternative(pdf_path, options, output_dir)

        except Exception as e:
            self.logger.error(f"PDF conversion failed: {str(e)}")
            raise ImageProcessingError(
                message=f"PDF conversion failed: {str(e)}",
                image_path=pdf_path,
                processing_operation="convert_with_library",
                original_exception=e,
            )

    def _convert_with_alternative(
        self, pdf_path: str, options: ImageProcessingOptions, output_dir: str
    ) -> List[str]:
        """
        대체 방법을 사용한 PDF 변환

        Args:
            pdf_path: PDF 파일 경로
            options: 변환 옵션
            output_dir: 출력 디렉터리

        Returns:
            List[str]: 변환된 이미지 파일 경로 목록
        """
        try:
            # pdf2image 사용 시도
            try:
                from pdf2image import convert_from_path

                images = convert_from_path(
                    pdf_path,
                    dpi=options.target_resolution,
                    fmt=options.output_format.value,
                )

                converted_images = []
                for i, image in enumerate(images):
                    output_filename = f"{Path(pdf_path).stem}_page_{i + 1:03d}.{options.output_format.value}"
                    output_path = os.path.join(output_dir, output_filename)

                    image.save(output_path, quality=options.output_quality)
                    converted_images.append(output_path)

                return converted_images

            except ImportError:
                self.logger.error("No PDF conversion library available")
                raise ImageProcessingError(
                    message="No PDF conversion library available (PyMuPDF or pdf2image required)",
                    image_path=pdf_path,
                    processing_operation="convert_with_alternative",
                )

        except Exception as e:
            self.logger.error(f"Alternative PDF conversion failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Alternative PDF conversion failed: {str(e)}",
                image_path=pdf_path,
                processing_operation="convert_with_alternative",
                original_exception=e,
            )

    def resize_image(
        self,
        image_path: str,
        target_size: Tuple[int, int],
        maintain_aspect_ratio: bool = True,
    ) -> str:
        """
        이미지 크기 조정

        Args:
            image_path: 이미지 파일 경로
            target_size: 목표 크기 (width, height)
            maintain_aspect_ratio: 종횡비 유지 여부

        Returns:
            str: 크기 조정된 이미지 경로
        """
        try:
            # PIL 사용
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    if maintain_aspect_ratio:
                        img.thumbnail(target_size, Image.Resampling.LANCZOS)
                    else:
                        img = img.resize(target_size, Image.Resampling.LANCZOS)

                    # 출력 경로 생성
                    output_path = self._generate_output_path(image_path, "_resized")

                    # 이미지 저장
                    img.save(output_path, quality=IMAGE_DEFAULT_QUALITY)

                    self.logger.debug(f"Image resized: {image_path} -> {output_path}")
                    return output_path

            except ImportError:
                self.logger.error("PIL not available for image resizing")
                raise ImageProcessingError(
                    message="PIL not available for image resizing",
                    image_path=image_path,
                    processing_operation="resize_image",
                )

        except Exception as e:
            self.logger.error(f"Image resizing failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Image resizing failed: {str(e)}",
                image_path=image_path,
                processing_operation="resize_image",
                original_exception=e,
            )

    def _generate_output_path(self, input_path: str, suffix: str) -> str:
        """
        출력 파일 경로 생성

        Args:
            input_path: 입력 파일 경로
            suffix: 접미사

        Returns:
            str: 출력 파일 경로
        """
        path_obj = Path(input_path)
        output_filename = f"{path_obj.stem}{suffix}{path_obj.suffix}"
        return str(path_obj.parent / output_filename)


# ====================================================================================
# 3. 이미지 개선 클래스
# ====================================================================================


class ImageEnhancer:
    """
    이미지 품질 개선 클래스

    이미지의 노이즈 제거, 선명도 향상, 대비 조정 등을 처리합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        ImageEnhancer 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        self.config = config
        self.logger = logger

    def enhance_image_quality(
        self, image_path: str, options: ImageProcessingOptions
    ) -> str:
        """
        이미지 품질 개선

        Args:
            image_path: 이미지 파일 경로
            options: 처리 옵션

        Returns:
            str: 개선된 이미지 경로
        """
        try:
            # 이미지 로드
            image_array = self._load_image_as_array(image_path)

            # 노이즈 제거
            if options.enable_noise_reduction:
                image_array = self._reduce_noise(image_array)

            # 선명도 향상
            if options.enable_sharpening:
                image_array = self._enhance_sharpness(image_array)

            # 대비 조정
            if options.enable_contrast_enhancement:
                image_array = self._enhance_contrast(image_array)

            # 밝기 조정
            if options.enable_brightness_adjustment:
                image_array = self._adjust_brightness(image_array)

            # 이미지 저장
            enhanced_path = self._save_enhanced_image(image_path, image_array)

            self.logger.info(f"Image enhanced: {image_path} -> {enhanced_path}")
            return enhanced_path

        except Exception as e:
            self.logger.error(f"Image enhancement failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Image enhancement failed: {str(e)}",
                image_path=image_path,
                processing_operation="enhance_image_quality",
                original_exception=e,
            )

    def correct_image_geometry(
        self, image_path: str, options: ImageProcessingOptions
    ) -> str:
        """
        이미지 기하학적 보정

        Args:
            image_path: 이미지 파일 경로
            options: 처리 옵션

        Returns:
            str: 보정된 이미지 경로
        """
        try:
            image_array = self._load_image_as_array(image_path)

            # 회전 보정
            if options.enable_rotation_correction:
                rotation_angle = self.detect_image_orientation(image_array)
                if abs(rotation_angle) > options.rotation_threshold:
                    image_array = self._rotate_image(image_array, -rotation_angle)

            # 기울기 보정
            if options.enable_skew_correction:
                skew_angle = self.detect_image_skew(image_array)
                if abs(skew_angle) > options.skew_threshold:
                    image_array = self._correct_skew(image_array, skew_angle)

            # 보정된 이미지 저장
            corrected_path = self._save_corrected_image(image_path, image_array)

            self.logger.info(
                f"Image geometry corrected: {image_path} -> {corrected_path}"
            )
            return corrected_path

        except Exception as e:
            self.logger.error(f"Image geometry correction failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Image geometry correction failed: {str(e)}",
                image_path=image_path,
                processing_operation="correct_image_geometry",
                original_exception=e,
            )

    def detect_image_orientation(self, image_array: np.ndarray) -> float:
        """
        이미지 회전 각도 감지

        Args:
            image_array: 이미지 배열

        Returns:
            float: 회전 각도 (도)
        """
        try:
            # OpenCV 사용
            try:
                import cv2

                # 그레이스케일 변환
                if len(image_array.shape) == 3:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array

                # 에지 검출
                edges = cv2.Canny(gray, 50, 150, apertureSize=3)

                # 허프 변환으로 직선 검출
                lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

                if lines is not None:
                    angles = []
                    for line in lines:
                        rho, theta = line[0]
                        angle = theta * 180 / np.pi
                        # 수직선과 수평선에 가까운 각도만 고려
                        if angle < 45 or angle > 135:
                            angles.append(angle)

                    if angles:
                        # 가장 빈번한 각도 계산
                        median_angle = np.median(angles)
                        if median_angle > 90:
                            median_angle = median_angle - 180
                        return median_angle

                return 0.0

            except ImportError:
                self.logger.warning("OpenCV not available for orientation detection")
                return 0.0

        except Exception as e:
            self.logger.error(f"Orientation detection failed: {str(e)}")
            return 0.0

    def detect_image_skew(self, image_array: np.ndarray) -> float:
        """
        이미지 기울기 감지

        Args:
            image_array: 이미지 배열

        Returns:
            float: 기울기 각도 (도)
        """
        try:
            # OpenCV 사용
            try:
                import cv2

                # 그레이스케일 변환
                if len(image_array.shape) == 3:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array

                # 이진화
                _, binary = cv2.threshold(
                    gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # 형태학적 연산으로 텍스트 라인 추출
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
                morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

                # 윤곽선 검출
                contours, _ = cv2.findContours(
                    morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                angles = []
                for contour in contours:
                    # 최소 회전 사각형 찾기
                    rect = cv2.minAreaRect(contour)
                    angle = rect[2]

                    # 각도 보정
                    if angle < -45:
                        angle = 90 + angle

                    angles.append(angle)

                if angles:
                    return np.median(angles)

                return 0.0

            except ImportError:
                self.logger.warning("OpenCV not available for skew detection")
                return 0.0

        except Exception as e:
            self.logger.error(f"Skew detection failed: {str(e)}")
            return 0.0

    def _load_image_as_array(self, image_path: str) -> np.ndarray:
        """이미지를 numpy 배열로 로드"""
        try:
            # PIL 사용
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    # RGB 형식으로 변환
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    return np.array(img)

            except ImportError:
                # OpenCV 사용
                try:
                    import cv2

                    img = cv2.imread(image_path)
                    if img is None:
                        raise ImageProcessingError(f"Cannot load image: {image_path}")

                    # BGR to RGB 변환
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                except ImportError:
                    raise ImageProcessingError("No image processing library available")

        except Exception as e:
            raise ImageProcessingError(f"Failed to load image: {str(e)}")

    def _reduce_noise(self, image_array: np.ndarray) -> np.ndarray:
        """노이즈 제거"""
        try:
            # OpenCV 사용
            try:
                import cv2

                # 비선형 필터 적용
                denoised = cv2.fastNlMeansDenoisingColored(
                    image_array, None, 10, 10, 7, 21
                )
                return denoised

            except ImportError:
                # 기본 가우시안 블러 적용
                from scipy import ndimage

                # 각 채널에 대해 가우시안 필터 적용
                if len(image_array.shape) == 3:
                    denoised = np.zeros_like(image_array)
                    for i in range(image_array.shape[2]):
                        denoised[:, :, i] = ndimage.gaussian_filter(
                            image_array[:, :, i], sigma=0.5
                        )
                    return denoised
                else:
                    return ndimage.gaussian_filter(image_array, sigma=0.5)

        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {str(e)}")
            return image_array

    def _enhance_sharpness(self, image_array: np.ndarray) -> np.ndarray:
        """선명도 향상"""
        try:
            # 언샵 마스킹 적용
            from scipy import ndimage

            # 가우시안 블러 적용
            if len(image_array.shape) == 3:
                blurred = np.zeros_like(image_array)
                for i in range(image_array.shape[2]):
                    blurred[:, :, i] = ndimage.gaussian_filter(
                        image_array[:, :, i], sigma=1.0
                    )
            else:
                blurred = ndimage.gaussian_filter(image_array, sigma=1.0)

            # 언샵 마스킹
            sharpened = image_array + IMAGE_SHARPENING_STRENGTH * (
                image_array - blurred
            )

            # 값 범위 조정
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

            return sharpened

        except Exception as e:
            self.logger.warning(f"Sharpening failed: {str(e)}")
            return image_array

    def _enhance_contrast(self, image_array: np.ndarray) -> np.ndarray:
        """대비 향상"""
        try:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization) 적용
            try:
                import cv2

                # LAB 색공간으로 변환
                lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)

                # L 채널에 CLAHE 적용
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])

                # RGB로 변환
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                return enhanced

            except ImportError:
                # 기본 히스토그램 평활화
                if len(image_array.shape) == 3:
                    # 각 채널에 대해 히스토그램 평활화
                    enhanced = np.zeros_like(image_array)
                    for i in range(image_array.shape[2]):
                        enhanced[:, :, i] = self._histogram_equalization(
                            image_array[:, :, i]
                        )
                    return enhanced
                else:
                    return self._histogram_equalization(image_array)

        except Exception as e:
            self.logger.warning(f"Contrast enhancement failed: {str(e)}")
            return image_array

    def _adjust_brightness(self, image_array: np.ndarray) -> np.ndarray:
        """밝기 조정"""
        try:
            # 이미지 평균 밝기 계산
            if len(image_array.shape) == 3:
                mean_brightness = np.mean(image_array)
            else:
                mean_brightness = np.mean(image_array)

            # 목표 밝기 (중간값)
            target_brightness = 128

            # 밝기 조정
            brightness_adjustment = target_brightness - mean_brightness

            # 조정값 제한
            brightness_adjustment = np.clip(brightness_adjustment, -50, 50)

            # 밝기 적용
            adjusted = image_array + brightness_adjustment
            adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)

            return adjusted

        except Exception as e:
            self.logger.warning(f"Brightness adjustment failed: {str(e)}")
            return image_array

    def _histogram_equalization(self, image_channel: np.ndarray) -> np.ndarray:
        """히스토그램 평활화"""
        try:
            # 히스토그램 계산
            hist, bins = np.histogram(image_channel.flatten(), 256, [0, 256])

            # 누적 분포 함수 계산
            cdf = hist.cumsum()
            cdf_normalized = cdf * 255 / cdf[-1]

            # 룩업 테이블 생성
            lut = np.interp(np.arange(256), bins[:-1], cdf_normalized)

            # 히스토그램 평활화 적용
            equalized = lut[image_channel]

            return equalized.astype(np.uint8)

        except Exception as e:
            self.logger.warning(f"Histogram equalization failed: {str(e)}")
            return image_channel

    def _rotate_image(self, image_array: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        try:
            # OpenCV 사용
            try:
                import cv2

                height, width = image_array.shape[:2]
                center = (width // 2, height // 2)

                # 회전 행렬 생성
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

                # 회전 적용
                rotated = cv2.warpAffine(
                    image_array,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

                return rotated

            except ImportError:
                # scipy 사용
                from scipy import ndimage

                rotated = ndimage.rotate(image_array, angle, reshape=False, cval=255)
                return rotated.astype(np.uint8)

        except Exception as e:
            self.logger.warning(f"Image rotation failed: {str(e)}")
            return image_array

    def _correct_skew(self, image_array: np.ndarray, skew_angle: float) -> np.ndarray:
        """기울기 보정"""
        try:
            # OpenCV 사용
            try:
                import cv2

                height, width = image_array.shape[:2]
                center = (width // 2, height // 2)

                # 회전 행렬 생성
                rotation_matrix = cv2.getRotationMatrix2D(center, -skew_angle, 1.0)

                # 회전 적용
                corrected = cv2.warpAffine(
                    image_array,
                    rotation_matrix,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )

                return corrected

            except ImportError:
                # scipy 사용
                from scipy import ndimage

                corrected = ndimage.rotate(
                    image_array, -skew_angle, reshape=False, cval=255
                )
                return corrected.astype(np.uint8)

        except Exception as e:
            self.logger.warning(f"Skew correction failed: {str(e)}")
            return image_array

    def _save_enhanced_image(self, original_path: str, image_array: np.ndarray) -> str:
        """개선된 이미지 저장"""
        try:
            # 출력 경로 생성
            path_obj = Path(original_path)
            enhanced_filename = f"{path_obj.stem}_enhanced{path_obj.suffix}"
            enhanced_path = str(path_obj.parent / enhanced_filename)

            # PIL 사용하여 저장
            try:
                from PIL import Image

                img = Image.fromarray(image_array)
                img.save(enhanced_path, quality=IMAGE_DEFAULT_QUALITY)

                return enhanced_path

            except ImportError:
                # OpenCV 사용
                import cv2

                # RGB to BGR 변환
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(enhanced_path, bgr_image)

                return enhanced_path

        except Exception as e:
            self.logger.error(f"Failed to save enhanced image: {str(e)}")
            raise ImageProcessingError(f"Failed to save enhanced image: {str(e)}")

    def _save_corrected_image(self, original_path: str, image_array: np.ndarray) -> str:
        """보정된 이미지 저장"""
        try:
            # 출력 경로 생성
            path_obj = Path(original_path)
            corrected_filename = f"{path_obj.stem}_corrected{path_obj.suffix}"
            corrected_path = str(path_obj.parent / corrected_filename)

            # PIL 사용하여 저장
            try:
                from PIL import Image

                img = Image.fromarray(image_array)
                img.save(corrected_path, quality=IMAGE_DEFAULT_QUALITY)

                return corrected_path

            except ImportError:
                # OpenCV 사용
                import cv2

                # RGB to BGR 변환
                bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                cv2.imwrite(corrected_path, bgr_image)

                return corrected_path

        except Exception as e:
            self.logger.error(f"Failed to save corrected image: {str(e)}")
            raise ImageProcessingError(f"Failed to save corrected image: {str(e)}")


# ====================================================================================
# 4. 이미지 검증 클래스
# ====================================================================================


class ImageValidator:
    """
    이미지 검증 클래스

    이미지 형식, 품질, 메타데이터 등을 검증합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        ImageValidator 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        self.config = config
        self.logger = logger

    def validate_image_format(self, image_path: str) -> bool:
        """
        이미지 형식 검증

        Args:
            image_path: 이미지 파일 경로

        Returns:
            bool: 검증 결과
        """
        try:
            if not os.path.exists(image_path):
                self.logger.error(f"Image file not found: {image_path}")
                return False

            # 파일 확장자 검증
            file_extension = Path(image_path).suffix.lower()
            if file_extension not in IMAGE_FILE_EXTENSIONS:
                self.logger.error(f"Unsupported image format: {file_extension}")
                return False

            # 이미지 파일 읽기 테스트
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    # 이미지 정보 확인
                    width, height = img.size

                    # 크기 검증
                    if width < IMAGE_MIN_DIMENSION or height < IMAGE_MIN_DIMENSION:
                        self.logger.error(f"Image too small: {width}x{height}")
                        return False

                    if width > IMAGE_MAX_DIMENSION or height > IMAGE_MAX_DIMENSION:
                        self.logger.error(f"Image too large: {width}x{height}")
                        return False

                    # 이미지 무결성 검증
                    img.verify()

                    return True

            except ImportError:
                self.logger.warning("PIL not available for image validation")
                return self._validate_with_opencv(image_path)

        except Exception as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            return False

    def _validate_with_opencv(self, image_path: str) -> bool:
        """OpenCV를 사용한 이미지 검증"""
        try:
            import cv2

            img = cv2.imread(image_path)
            if img is None:
                return False

            height, width = img.shape[:2]

            # 크기 검증
            if width < IMAGE_MIN_DIMENSION or height < IMAGE_MIN_DIMENSION:
                return False

            if width > IMAGE_MAX_DIMENSION or height > IMAGE_MAX_DIMENSION:
                return False

            return True

        except ImportError:
            self.logger.error("No image processing library available for validation")
            return False
        except Exception as e:
            self.logger.error(f"OpenCV validation failed: {str(e)}")
            return False

    def assess_image_quality(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 품질 평가

        Args:
            image_path: 이미지 파일 경로

        Returns:
            Dict[str, Any]: 품질 평가 결과
        """
        try:
            # 이미지 로드
            image_array = self._load_image_for_assessment(image_path)

            # 품질 메트릭 계산
            quality_metrics = {
                "sharpness": self._calculate_sharpness(image_array),
                "contrast": self._calculate_contrast(image_array),
                "brightness": self._calculate_brightness(image_array),
                "noise_level": self._calculate_noise_level(image_array),
                "overall_quality": 0.0,
            }

            # 전체 품질 점수 계산
            quality_metrics["overall_quality"] = self._calculate_overall_quality(
                quality_metrics
            )

            # 품질 등급 결정
            quality_metrics["quality_grade"] = self._determine_quality_grade(
                quality_metrics["overall_quality"]
            )

            return quality_metrics

        except Exception as e:
            self.logger.error(f"Image quality assessment failed: {str(e)}")
            return {
                "error": str(e),
                "overall_quality": 0.0,
                "quality_grade": ImageQuality.LOW.value,
            }

    def _load_image_for_assessment(self, image_path: str) -> np.ndarray:
        """품질 평가용 이미지 로드"""
        try:
            # PIL 사용
            try:
                from PIL import Image

                with Image.open(image_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    return np.array(img)

            except ImportError:
                # OpenCV 사용
                import cv2

                img = cv2.imread(image_path)
                if img is None:
                    raise ImageProcessingError(f"Cannot load image: {image_path}")

                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        except Exception as e:
            raise ImageProcessingError(f"Failed to load image for assessment: {str(e)}")

    def _calculate_sharpness(self, image_array: np.ndarray) -> float:
        """선명도 계산"""
        try:
            # 그레이스케일 변환
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array

            # 라플라시안 분산 계산
            try:
                import cv2

                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = laplacian.var()
            except ImportError:
                # 기본 라플라시안 커널 사용
                laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
                from scipy import ndimage

                laplacian = ndimage.convolve(gray, laplacian_kernel)
                sharpness = laplacian.var()

            return float(sharpness)

        except Exception as e:
            self.logger.warning(f"Sharpness calculation failed: {str(e)}")
            return 0.0

    def _calculate_contrast(self, image_array: np.ndarray) -> float:
        """대비 계산"""
        try:
            # 그레이스케일 변환
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array

            # RMS 대비 계산
            contrast = np.std(gray)
            return float(contrast)

        except Exception as e:
            self.logger.warning(f"Contrast calculation failed: {str(e)}")
            return 0.0

    def _calculate_brightness(self, image_array: np.ndarray) -> float:
        """밝기 계산"""
        try:
            # 평균 밝기 계산
            if len(image_array.shape) == 3:
                brightness = np.mean(image_array)
            else:
                brightness = np.mean(image_array)

            return float(brightness)

        except Exception as e:
            self.logger.warning(f"Brightness calculation failed: {str(e)}")
            return 0.0

    def _calculate_noise_level(self, image_array: np.ndarray) -> float:
        """노이즈 레벨 계산"""
        try:
            # 그레이스케일 변환
            if len(image_array.shape) == 3:
                gray = np.mean(image_array, axis=2).astype(np.uint8)
            else:
                gray = image_array

            # 가우시안 블러 적용
            from scipy import ndimage

            blurred = ndimage.gaussian_filter(gray, sigma=1.0)

            # 노이즈 추정 (원본과 블러된 이미지의 차이)
            noise = np.abs(gray.astype(float) - blurred.astype(float))
            noise_level = np.mean(noise)

            return float(noise_level)

        except Exception as e:
            self.logger.warning(f"Noise level calculation failed: {str(e)}")
            return 0.0

    def _calculate_overall_quality(self, metrics: Dict[str, float]) -> float:
        """전체 품질 점수 계산"""
        try:
            # 가중치 설정
            weights = {
                "sharpness": 0.3,
                "contrast": 0.3,
                "brightness": 0.2,
                "noise_level": 0.2,
            }

            # 정규화된 점수 계산
            sharpness_score = min(metrics["sharpness"] / 1000, 1.0)  # 선명도 정규화
            contrast_score = min(metrics["contrast"] / 100, 1.0)  # 대비 정규화
            brightness_score = (
                1.0 - abs(metrics["brightness"] - 128) / 128
            )  # 밝기 정규화
            noise_score = max(0, 1.0 - metrics["noise_level"] / 50)  # 노이즈 정규화

            # 가중 평균 계산
            overall_quality = (
                weights["sharpness"] * sharpness_score
                + weights["contrast"] * contrast_score
                + weights["brightness"] * brightness_score
                + weights["noise_level"] * noise_score
            )

            return float(overall_quality)

        except Exception as e:
            self.logger.warning(f"Overall quality calculation failed: {str(e)}")
            return 0.0

    def _determine_quality_grade(self, overall_quality: float) -> str:
        """품질 등급 결정"""
        if overall_quality >= 0.8:
            return ImageQuality.ULTRA.value
        elif overall_quality >= 0.6:
            return ImageQuality.HIGH.value
        elif overall_quality >= 0.4:
            return ImageQuality.MEDIUM.value
        else:
            return ImageQuality.LOW.value


# ====================================================================================
# 5. 유틸리티 함수들
# ====================================================================================


def convert_pdf_to_images(
    pdf_path: str, output_dir: str, options: Optional[ImageProcessingOptions] = None
) -> List[str]:
    """
    PDF를 이미지로 변환하는 유틸리티 함수

    Args:
        pdf_path: PDF 파일 경로
        output_dir: 출력 디렉터리
        options: 변환 옵션

    Returns:
        List[str]: 변환된 이미지 파일 경로 목록
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_converter")

    converter = ImageConverter(config, logger)

    if options is None:
        options = ImageProcessingOptions()

    return converter.convert_pdf_to_images(pdf_path, options, output_dir)


def resize_image(image_path: str, target_size: Tuple[int, int]) -> str:
    """
    이미지 크기 조정 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로
        target_size: 목표 크기 (width, height)

    Returns:
        str: 크기 조정된 이미지 경로
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_converter")

    converter = ImageConverter(config, logger)
    return converter.resize_image(image_path, target_size)


def enhance_image_quality(
    image_path: str, options: Optional[ImageProcessingOptions] = None
) -> str:
    """
    이미지 품질 개선 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로
        options: 처리 옵션

    Returns:
        str: 개선된 이미지 경로
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_enhancer")

    enhancer = ImageEnhancer(config, logger)

    if options is None:
        options = ImageProcessingOptions()

    return enhancer.enhance_image_quality(image_path, options)


def detect_image_orientation(image_path: str) -> float:
    """
    이미지 회전 각도 감지 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로

    Returns:
        float: 회전 각도 (도)
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_enhancer")

    enhancer = ImageEnhancer(config, logger)

    # 이미지 로드
    image_array = enhancer._load_image_as_array(image_path)

    return enhancer.detect_image_orientation(image_array)


def correct_image_skew(image_path: str) -> str:
    """
    이미지 기울기 보정 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로

    Returns:
        str: 보정된 이미지 경로
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_enhancer")

    enhancer = ImageEnhancer(config, logger)

    options = ImageProcessingOptions()
    options.enable_skew_correction = True

    return enhancer.correct_image_geometry(image_path, options)


def validate_image_format(image_path: str) -> bool:
    """
    이미지 형식 검증 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로

    Returns:
        bool: 검증 결과
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_validator")

    validator = ImageValidator(config, logger)
    return validator.validate_image_format(image_path)


def assess_image_quality(image_path: str) -> Dict[str, Any]:
    """
    이미지 품질 평가 유틸리티 함수

    Args:
        image_path: 이미지 파일 경로

    Returns:
        Dict[str, Any]: 품질 평가 결과
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_validator")

    validator = ImageValidator(config, logger)
    return validator.assess_image_quality(image_path)


def create_image_processing_options(**kwargs) -> ImageProcessingOptions:
    """
    이미지 처리 옵션 생성 유틸리티 함수

    Args:
        **kwargs: 옵션 키워드 인수

    Returns:
        ImageProcessingOptions: 생성된 옵션 객체
    """
    options = ImageProcessingOptions()

    # 옵션 설정
    for key, value in kwargs.items():
        if hasattr(options, key):
            setattr(options, key, value)

    return options


# ====================================================================================
# 6. 배치 처리 함수
# ====================================================================================


def process_image_batch(
    image_paths: List[str],
    output_dir: str,
    options: Optional[ImageProcessingOptions] = None,
) -> List[Dict[str, Any]]:
    """
    이미지 배치 처리

    Args:
        image_paths: 이미지 파일 경로 목록
        output_dir: 출력 디렉터리
        options: 처리 옵션

    Returns:
        List[Dict[str, Any]]: 처리 결과 목록
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_processor")

    processor = ImageProcessor(config, logger)

    if options is None:
        options = ImageProcessingOptions()

    results = []

    for image_path in image_paths:
        try:
            # 처리 데이터 준비
            processing_data = {
                "file_path": image_path,
                "output_dir": output_dir,
                **options.__dict__,
            }

            # 이미지 처리
            result = processor.process(processing_data)
            results.append(result)

        except Exception as e:
            logger.error(f"Batch processing failed for {image_path}: {str(e)}")
            results.append(
                {"file_path": image_path, "error": str(e), "processing_success": False}
            )

    return results


def convert_pdf_batch(
    pdf_paths: List[str],
    output_dir: str,
    options: Optional[ImageProcessingOptions] = None,
) -> List[Dict[str, Any]]:
    """
    PDF 배치 변환

    Args:
        pdf_paths: PDF 파일 경로 목록
        output_dir: 출력 디렉터리
        options: 변환 옵션

    Returns:
        List[Dict[str, Any]]: 변환 결과 목록
    """
    from config.settings import load_configuration

    config = load_configuration()
    logger = get_application_logger("image_converter")

    converter = ImageConverter(config, logger)

    if options is None:
        options = ImageProcessingOptions()

    results = []

    for pdf_path in pdf_paths:
        try:
            converted_images = converter.convert_pdf_to_images(
                pdf_path, options, output_dir
            )

            results.append(
                {
                    "pdf_path": pdf_path,
                    "converted_images": converted_images,
                    "total_pages": len(converted_images),
                    "conversion_success": True,
                }
            )

        except Exception as e:
            logger.error(f"PDF conversion failed for {pdf_path}: {str(e)}")
            results.append(
                {"pdf_path": pdf_path, "error": str(e), "conversion_success": False}
            )

    return results


# ====================================================================================
# 7. 런타임 검증 및 테스트
# ====================================================================================


def validate_image_processing_libraries() -> Dict[str, bool]:
    """
    이미지 처리 라이브러리 가용성 검증

    Returns:
        Dict[str, bool]: 라이브러리 가용성 상태
    """
    libraries = {
        "PIL": False,
        "OpenCV": False,
        "PyMuPDF": False,
        "pdf2image": False,
        "scipy": False,
        "numpy": False,
    }

    # PIL 검증
    try:
        from PIL import Image

        libraries["PIL"] = True
    except ImportError:
        pass

    # OpenCV 검증
    try:
        import cv2

        libraries["OpenCV"] = True
    except ImportError:
        pass

    # PyMuPDF 검증
    try:
        import fitz

        libraries["PyMuPDF"] = True
    except ImportError:
        pass

    # pdf2image 검증
    try:
        from pdf2image import convert_from_path

        libraries["pdf2image"] = True
    except ImportError:
        pass

    # scipy 검증
    try:
        from scipy import ndimage

        libraries["scipy"] = True
    except ImportError:
        pass

    # numpy 검증
    try:
        import numpy as np

        libraries["numpy"] = True
    except ImportError:
        pass

    return libraries


if __name__ == "__main__":
    # 이미지 처리 유틸리티 테스트
    print("YOKOGAWA OCR 이미지 처리 유틸리티 테스트")
    print("=" * 50)

    try:
        # 라이브러리 가용성 검증
        libraries = validate_image_processing_libraries()
        print("📚 라이브러리 가용성 검증:")
        for lib, available in libraries.items():
            status = "✅" if available else "❌"
            print(f"  {status} {lib}: {'사용 가능' if available else '사용 불가'}")

        # 기본 설정으로 테스트
        from config.settings import load_configuration

        config = load_configuration()
        logger = get_application_logger("image_processor_test")

        # 이미지 프로세서 생성
        processor = ImageProcessor(config, logger)
        print("✅ ImageProcessor 생성 완료")

        # 처리 옵션 생성
        options = create_image_processing_options(
            output_format=ImageFormat.PNG,
            enable_noise_reduction=True,
            enable_sharpening=True,
        )
        print("✅ 처리 옵션 생성 완료")

        # 임시 테스트 이미지 생성 (실제 사용 시에는 실제 파일 경로 사용)
        test_image_path = "test_image.png"

        # 테스트 이미지가 없으면 테스트 스킵
        if not os.path.exists(test_image_path):
            print(
                "⚠️  테스트 이미지가 없습니다. 실제 이미지 파일을 사용하여 테스트하세요."
            )
        else:
            # 이미지 검증 테스트
            is_valid = validate_image_format(test_image_path)
            print(f"✅ 이미지 형식 검증: {'통과' if is_valid else '실패'}")

            # 이미지 품질 평가 테스트
            quality_result = assess_image_quality(test_image_path)
            print(
                f"✅ 품질 평가 완료: {quality_result.get('quality_grade', 'unknown')}"
            )

        # 처리 통계 출력
        stats = processor.get_processing_statistics()
        print(f"📊 처리 통계: {stats}")

    except Exception as e:
        print(f"❌ 이미지 처리 유틸리티 테스트 실패: {e}")

    print("\n🎯 이미지 처리 유틸리티 구현이 완료되었습니다!")
