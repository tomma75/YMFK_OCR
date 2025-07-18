#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 데이터 증강 서비스 모듈

이 모듈은 문서 및 이미지 데이터를 증강하는 서비스를 제공합니다.
기하학적 변환, 색상 조정, 노이즈 추가 등의 증강 기법을 지원합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import json
import uuid
import random
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from PIL.ImageDraw import ImageDraw
import cv2

from core.base_classes import BaseService, AugmentationInterface
from core.exceptions import (
    AugmentationError,
    ImageProcessingError,
    ProcessingError,
    ValidationError,
    FileAccessError,
    ApplicationError,
)
from config.settings import ApplicationConfig
from config.constants import (
    DEFAULT_AUGMENTATION_FACTOR,
    MAX_AUGMENTATION_FACTOR,
    MIN_AUGMENTATION_FACTOR,
    GEOMETRIC_ROTATION_ANGLES,
    GEOMETRIC_SCALE_FACTORS,
    GEOMETRIC_TRANSLATION_RANGE,
    GEOMETRIC_SHEAR_RANGE,
    COLOR_BRIGHTNESS_DELTA,
    COLOR_CONTRAST_DELTA,
    COLOR_SATURATION_DELTA,
    COLOR_HUE_DELTA,
    NOISE_GAUSSIAN_MEAN,
    NOISE_GAUSSIAN_STD,
    NOISE_SALT_PEPPER_AMOUNT,
    NOISE_SPECKLE_VARIANCE,
    AUGMENTATION_ROTATION_RANGE,
    AUGMENTATION_SCALE_RANGE,
    AUGMENTATION_BRIGHTNESS_RANGE,
    AUGMENTATION_CONTRAST_RANGE,
    AUGMENTATION_NOISE_VARIANCE,
    DEFAULT_BATCH_SIZE,
    MAX_WORKER_THREADS,
)
from models.document_model import DocumentModel, DocumentStatus
from models.annotation_model import AnnotationModel, BoundingBox
from utils.logger_util import get_application_logger
from utils.file_handler import FileHandler


class ImageAugmenter:
    """
    이미지 증강 클래스

    다양한 이미지 증강 기법을 제공합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        ImageAugmenter 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("image_augmenter")

        # 증강 파라미터 설정
        self.rotation_range = AUGMENTATION_ROTATION_RANGE
        self.scale_range = AUGMENTATION_SCALE_RANGE
        self.brightness_range = AUGMENTATION_BRIGHTNESS_RANGE
        self.contrast_range = AUGMENTATION_CONTRAST_RANGE
        self.noise_variance = AUGMENTATION_NOISE_VARIANCE

        # 통계 정보
        self.images_processed = 0
        self.images_generated = 0
        self.processing_errors = 0

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

        self.logger.info("ImageAugmenter initialized")

    def augment_image(
        self, image_path: str, augmentation_types: List[str]
    ) -> List[str]:
        """
        이미지 증강 수행

        Args:
            image_path: 원본 이미지 경로
            augmentation_types: 적용할 증강 기법 목록

        Returns:
            List[str]: 증강된 이미지 파일 경로 목록
        """
        try:
            if not os.path.exists(image_path):
                raise FileAccessError(
                    message=f"Image file not found: {image_path}",
                    file_path=image_path,
                    access_type="read",
                )

            # 원본 이미지 로드
            original_image = Image.open(image_path)
            augmented_images = []

            # 증강 기법별 처리
            for aug_type in augmentation_types:
                try:
                    if aug_type == "rotation":
                        rotated_images = self._apply_rotation(original_image)
                        augmented_images.extend(rotated_images)
                    elif aug_type == "scaling":
                        scaled_images = self._apply_scaling(original_image)
                        augmented_images.extend(scaled_images)
                    elif aug_type == "brightness":
                        bright_images = self._apply_brightness(original_image)
                        augmented_images.extend(bright_images)
                    elif aug_type == "contrast":
                        contrast_images = self._apply_contrast(original_image)
                        augmented_images.extend(contrast_images)
                    elif aug_type == "noise":
                        noisy_images = self._apply_noise(original_image)
                        augmented_images.extend(noisy_images)
                    else:
                        self.logger.warning(f"Unknown augmentation type: {aug_type}")

                except Exception as e:
                    self.logger.error(
                        f"Failed to apply {aug_type} augmentation: {str(e)}"
                    )
                    with self._lock:
                        self.processing_errors += 1

            # 증강된 이미지 저장
            saved_paths = self._save_augmented_images(augmented_images, image_path)

            with self._lock:
                self.images_processed += 1
                self.images_generated += len(saved_paths)

            return saved_paths

        except Exception as e:
            self.logger.error(f"Image augmentation failed for {image_path}: {str(e)}")
            with self._lock:
                self.processing_errors += 1
            raise ImageProcessingError(
                message=f"Image augmentation failed: {str(e)}",
                image_path=image_path,
                processing_operation="augment_image",
                original_exception=e,
            )

    def _apply_rotation(self, image: Image.Image) -> List[Image.Image]:
        """회전 증강 적용"""
        rotated_images = []

        for angle in GEOMETRIC_ROTATION_ANGLES:
            try:
                rotated = image.rotate(angle, expand=True, fillcolor="white")
                rotated_images.append(rotated)
            except Exception as e:
                self.logger.warning(f"Rotation failed for angle {angle}: {str(e)}")

        return rotated_images

    def _apply_scaling(self, image: Image.Image) -> List[Image.Image]:
        """크기 조정 증강 적용"""
        scaled_images = []
        original_size = image.size

        for scale_factor in GEOMETRIC_SCALE_FACTORS:
            try:
                new_size = (
                    int(original_size[0] * scale_factor),
                    int(original_size[1] * scale_factor),
                )
                scaled = image.resize(new_size, Image.LANCZOS)

                # 원본 크기로 복원 (패딩 또는 크롭)
                if scale_factor > 1.0:
                    # 크롭
                    left = (scaled.width - original_size[0]) // 2
                    top = (scaled.height - original_size[1]) // 2
                    right = left + original_size[0]
                    bottom = top + original_size[1]
                    scaled = scaled.crop((left, top, right, bottom))
                else:
                    # 패딩
                    scaled = ImageOps.pad(scaled, original_size, color="white")

                scaled_images.append(scaled)
            except Exception as e:
                self.logger.warning(
                    f"Scaling failed for factor {scale_factor}: {str(e)}"
                )

        return scaled_images

    def _apply_brightness(self, image: Image.Image) -> List[Image.Image]:
        """밝기 조정 증강 적용"""
        bright_images = []

        brightness_factors = [
            self.brightness_range[0],
            (self.brightness_range[0] + self.brightness_range[1]) / 2,
            self.brightness_range[1],
        ]

        for factor in brightness_factors:
            try:
                enhancer = ImageEnhance.Brightness(image)
                bright_image = enhancer.enhance(factor)
                bright_images.append(bright_image)
            except Exception as e:
                self.logger.warning(
                    f"Brightness adjustment failed for factor {factor}: {str(e)}"
                )

        return bright_images

    def _apply_contrast(self, image: Image.Image) -> List[Image.Image]:
        """대비 조정 증강 적용"""
        contrast_images = []

        contrast_factors = [
            self.contrast_range[0],
            (self.contrast_range[0] + self.contrast_range[1]) / 2,
            self.contrast_range[1],
        ]

        for factor in contrast_factors:
            try:
                enhancer = ImageEnhance.Contrast(image)
                contrast_image = enhancer.enhance(factor)
                contrast_images.append(contrast_image)
            except Exception as e:
                self.logger.warning(
                    f"Contrast adjustment failed for factor {factor}: {str(e)}"
                )

        return contrast_images

    def _apply_noise(self, image: Image.Image) -> List[Image.Image]:
        """노이즈 추가 증강 적용"""
        noisy_images = []

        try:
            # 가우시안 노이즈
            gaussian_noisy = self._add_gaussian_noise(image)
            noisy_images.append(gaussian_noisy)

            # 소금 후추 노이즈
            salt_pepper_noisy = self._add_salt_pepper_noise(image)
            noisy_images.append(salt_pepper_noisy)

        except Exception as e:
            self.logger.warning(f"Noise addition failed: {str(e)}")

        return noisy_images

    def _add_gaussian_noise(self, image: Image.Image) -> Image.Image:
        """가우시안 노이즈 추가"""
        img_array = np.array(image)

        # 가우시안 노이즈 생성
        noise = np.random.normal(
            NOISE_GAUSSIAN_MEAN, NOISE_GAUSSIAN_STD * 255, img_array.shape
        )

        # 노이즈 추가
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy_array)

    def _add_salt_pepper_noise(self, image: Image.Image) -> Image.Image:
        """소금 후추 노이즈 추가"""
        img_array = np.array(image)

        # 소금 후추 노이즈 생성
        noise = np.random.random(img_array.shape[:2])

        # 소금 노이즈 (흰색)
        salt_mask = noise < NOISE_SALT_PEPPER_AMOUNT / 2
        img_array[salt_mask] = 255

        # 후추 노이즈 (검은색)
        pepper_mask = noise > 1 - NOISE_SALT_PEPPER_AMOUNT / 2
        img_array[pepper_mask] = 0

        return Image.fromarray(img_array)

    def _save_augmented_images(
        self, images: List[Image.Image], original_path: str
    ) -> List[str]:
        """증강된 이미지 저장"""
        saved_paths = []
        base_name = Path(original_path).stem
        base_dir = Path(original_path).parent

        for i, img in enumerate(images):
            try:
                # 저장 경로 생성
                save_path = base_dir / f"{base_name}_aug_{i:03d}.jpg"

                # 이미지 저장
                img.save(str(save_path), "JPEG", quality=95)
                saved_paths.append(str(save_path))

            except Exception as e:
                self.logger.error(f"Failed to save augmented image {i}: {str(e)}")

        return saved_paths

    def get_statistics(self) -> Dict[str, Any]:
        """증강 통계 정보 반환"""
        with self._lock:
            return {
                "images_processed": self.images_processed,
                "images_generated": self.images_generated,
                "processing_errors": self.processing_errors,
                "generation_ratio": (
                    self.images_generated / self.images_processed
                    if self.images_processed > 0
                    else 0
                ),
            }


class GeometricTransformer:
    """
    기하학적 변환 클래스

    회전, 크기 조정, 전단 변환 등을 처리합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        GeometricTransformer 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("geometric_transformer")

        # 변환 파라미터
        self.rotation_angles = GEOMETRIC_ROTATION_ANGLES
        self.scale_factors = GEOMETRIC_SCALE_FACTORS
        self.translation_range = GEOMETRIC_TRANSLATION_RANGE
        self.shear_range = GEOMETRIC_SHEAR_RANGE

        self.logger.info("GeometricTransformer initialized")

    def apply_geometric_transformations(self, image: np.ndarray) -> List[np.ndarray]:
        """
        기하학적 변환 적용

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 변환된 이미지 목록
        """
        try:
            transformed_images = []

            # 회전 변환
            for angle in self.rotation_angles:
                rotated = self._rotate_image(image, angle)
                transformed_images.append(rotated)

            # 크기 조정
            for scale in self.scale_factors:
                scaled = self._scale_image(image, scale)
                transformed_images.append(scaled)

            # 전단 변환
            sheared = self._shear_image(image, self.shear_range)
            transformed_images.append(sheared)

            # 평행 이동
            translated = self._translate_image(image, self.translation_range)
            transformed_images.append(translated)

            return transformed_images

        except Exception as e:
            self.logger.error(f"Geometric transformation failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Geometric transformation failed: {str(e)}",
                processing_operation="apply_geometric_transformations",
                original_exception=e,
            )

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """이미지 회전"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        # 회전 매트릭스 생성
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 회전 적용
        rotated = cv2.warpAffine(
            image, rotation_matrix, (w, h), borderValue=(255, 255, 255)
        )

        return rotated

    def _scale_image(self, image: np.ndarray, scale_factor: float) -> np.ndarray:
        """이미지 크기 조정"""
        h, w = image.shape[:2]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)

        # 크기 조정
        scaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 원본 크기로 조정 (패딩 또는 크롭)
        if scale_factor > 1.0:
            # 크롭
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            scaled = scaled[start_h : start_h + h, start_w : start_w + w]
        else:
            # 패딩
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            scaled = cv2.copyMakeBorder(
                scaled,
                pad_h,
                h - new_h - pad_h,
                pad_w,
                w - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255),
            )

        return scaled

    def _shear_image(self, image: np.ndarray, shear_factor: float) -> np.ndarray:
        """이미지 전단 변환"""
        h, w = image.shape[:2]

        # 전단 매트릭스 생성
        shear_matrix = np.float32([[1, shear_factor, 0], [0, 1, 0]])

        # 전단 변환 적용
        sheared = cv2.warpAffine(
            image, shear_matrix, (w, h), borderValue=(255, 255, 255)
        )

        return sheared

    def _translate_image(self, image: np.ndarray, translation_range: int) -> np.ndarray:
        """이미지 평행 이동"""
        h, w = image.shape[:2]

        # 랜덤 이동 거리 생성
        dx = random.randint(-translation_range, translation_range)
        dy = random.randint(-translation_range, translation_range)

        # 이동 매트릭스 생성
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

        # 평행 이동 적용
        translated = cv2.warpAffine(
            image, translation_matrix, (w, h), borderValue=(255, 255, 255)
        )

        return translated


class ColorAdjuster:
    """
    색상 조정 클래스

    밝기, 대비, 채도, 색조 조정을 처리합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        ColorAdjuster 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("color_adjuster")

        # 색상 조정 파라미터
        self.brightness_delta = COLOR_BRIGHTNESS_DELTA
        self.contrast_delta = COLOR_CONTRAST_DELTA
        self.saturation_delta = COLOR_SATURATION_DELTA
        self.hue_delta = COLOR_HUE_DELTA

        self.logger.info("ColorAdjuster initialized")

    def apply_color_adjustments(self, image: np.ndarray) -> List[np.ndarray]:
        """
        색상 조정 적용

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 색상 조정된 이미지 목록
        """
        try:
            adjusted_images = []

            # PIL 이미지로 변환
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 밝기 조정
            brightness_adjusted = self._adjust_brightness(pil_image)
            adjusted_images.extend(brightness_adjusted)

            # 대비 조정
            contrast_adjusted = self._adjust_contrast(pil_image)
            adjusted_images.extend(contrast_adjusted)

            # 채도 조정
            saturation_adjusted = self._adjust_saturation(pil_image)
            adjusted_images.extend(saturation_adjusted)

            # 색조 조정
            hue_adjusted = self._adjust_hue(pil_image)
            adjusted_images.extend(hue_adjusted)

            # numpy 배열로 변환
            numpy_images = []
            for img in adjusted_images:
                numpy_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                numpy_images.append(numpy_img)

            return numpy_images

        except Exception as e:
            self.logger.error(f"Color adjustment failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Color adjustment failed: {str(e)}",
                processing_operation="apply_color_adjustments",
                original_exception=e,
            )

    def _adjust_brightness(self, image: Image.Image) -> List[Image.Image]:
        """밝기 조정"""
        adjusted_images = []

        brightness_factors = [1.0 - self.brightness_delta, 1.0 + self.brightness_delta]

        for factor in brightness_factors:
            enhancer = ImageEnhance.Brightness(image)
            adjusted = enhancer.enhance(factor)
            adjusted_images.append(adjusted)

        return adjusted_images

    def _adjust_contrast(self, image: Image.Image) -> List[Image.Image]:
        """대비 조정"""
        adjusted_images = []

        contrast_factors = [1.0 - self.contrast_delta, 1.0 + self.contrast_delta]

        for factor in contrast_factors:
            enhancer = ImageEnhance.Contrast(image)
            adjusted = enhancer.enhance(factor)
            adjusted_images.append(adjusted)

        return adjusted_images

    def _adjust_saturation(self, image: Image.Image) -> List[Image.Image]:
        """채도 조정"""
        adjusted_images = []

        saturation_factors = [1.0 - self.saturation_delta, 1.0 + self.saturation_delta]

        for factor in saturation_factors:
            enhancer = ImageEnhance.Color(image)
            adjusted = enhancer.enhance(factor)
            adjusted_images.append(adjusted)

        return adjusted_images

    def _adjust_hue(self, image: Image.Image) -> List[Image.Image]:
        """색조 조정"""
        adjusted_images = []

        try:
            # HSV 변환을 통한 색조 조정
            hsv_image = image.convert("HSV")
            h, s, v = hsv_image.split()

            # 색조 조정
            hue_adjustments = [-self.hue_delta, self.hue_delta]

            for adjustment in hue_adjustments:
                h_array = np.array(h)
                h_array = (h_array + adjustment * 255) % 256

                adjusted_h = Image.fromarray(h_array.astype(np.uint8), mode="L")
                adjusted_hsv = Image.merge("HSV", [adjusted_h, s, v])
                adjusted_rgb = adjusted_hsv.convert("RGB")

                adjusted_images.append(adjusted_rgb)

        except Exception as e:
            self.logger.warning(f"Hue adjustment failed: {str(e)}")

        return adjusted_images


class NoiseGenerator:
    """
    노이즈 생성 클래스

    다양한 종류의 노이즈를 생성하고 이미지에 추가합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        NoiseGenerator 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("noise_generator")

        # 노이즈 파라미터
        self.gaussian_mean = NOISE_GAUSSIAN_MEAN
        self.gaussian_std = NOISE_GAUSSIAN_STD
        self.salt_pepper_amount = NOISE_SALT_PEPPER_AMOUNT
        self.speckle_variance = NOISE_SPECKLE_VARIANCE

        self.logger.info("NoiseGenerator initialized")

    def add_noise_variations(self, image: np.ndarray) -> List[np.ndarray]:
        """
        노이즈 변형 추가

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 노이즈가 추가된 이미지 목록
        """
        try:
            noisy_images = []

            # 가우시안 노이즈
            gaussian_noisy = self._add_gaussian_noise(image)
            noisy_images.append(gaussian_noisy)

            # 소금 후추 노이즈
            salt_pepper_noisy = self._add_salt_pepper_noise(image)
            noisy_images.append(salt_pepper_noisy)

            # 스페클 노이즈
            speckle_noisy = self._add_speckle_noise(image)
            noisy_images.append(speckle_noisy)

            return noisy_images

        except Exception as e:
            self.logger.error(f"Noise generation failed: {str(e)}")
            raise ImageProcessingError(
                message=f"Noise generation failed: {str(e)}",
                processing_operation="add_noise_variations",
                original_exception=e,
            )

    def _add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """가우시안 노이즈 추가"""
        noise = np.random.normal(
            self.gaussian_mean, self.gaussian_std * 255, image.shape
        )
        noisy_image = np.clip(image + noise, 0, 255).astype(np.uint8)
        return noisy_image

    def _add_salt_pepper_noise(self, image: np.ndarray) -> np.ndarray:
        """소금 후추 노이즈 추가"""
        noisy_image = image.copy()

        # 소금 노이즈 (흰색)
        salt_mask = np.random.random(image.shape[:2]) < self.salt_pepper_amount / 2
        noisy_image[salt_mask] = 255

        # 후추 노이즈 (검은색)
        pepper_mask = (
            np.random.random(image.shape[:2]) > 1 - self.salt_pepper_amount / 2
        )
        noisy_image[pepper_mask] = 0

        return noisy_image

    def _add_speckle_noise(self, image: np.ndarray) -> np.ndarray:
        """스페클 노이즈 추가"""
        noise = np.random.normal(0, self.speckle_variance, image.shape)
        noise = image + image * noise
        noisy_image = np.clip(noise, 0, 255).astype(np.uint8)
        return noisy_image


class AugmentationService(BaseService, AugmentationInterface):
    """
    데이터 증강 서비스 클래스

    문서 및 이미지 데이터의 증강을 담당하는 메인 서비스입니다.
    BaseService와 AugmentationInterface를 구현합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        AugmentationService 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)

        # 컴포넌트 초기화
        self.file_handler = FileHandler(config)
        self.image_augmenter = ImageAugmenter(config)
        self.geometric_transformer = GeometricTransformer(config)
        self.color_adjuster = ColorAdjuster(config)
        self.noise_generator = NoiseGenerator(config)

        # 증강 설정
        self.augmentation_factor = config.processing_config.augmentation_factor
        self.max_workers = config.processing_config.max_workers
        self.batch_size = config.processing_config.batch_size

        # 증강 규칙
        self.augmentation_rules: Dict[str, Any] = {
            "enabled_techniques": [
                "rotation",
                "scaling",
                "brightness",
                "contrast",
                "noise",
            ],
            "geometric_enabled": True,
            "color_enabled": True,
            "noise_enabled": True,
            "preserve_aspect_ratio": True,
            "maintain_readability": True,
        }

        # 상태 관리
        self.source_documents: List[DocumentModel] = []
        self.augmented_documents: List[DocumentModel] = []
        self.augmentation_progress: float = 0.0
        self.current_operation: Optional[str] = None
        self.processing_errors: List[str] = []

        # 통계 정보
        self.augmentation_statistics: Dict[str, Any] = {}

        # 콜백 관리
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """
        서비스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("Initializing AugmentationService")

            # 상태 초기화
            with self._lock:
                self.source_documents.clear()
                self.augmented_documents.clear()
                self.augmentation_statistics.clear()
                self.processing_errors.clear()
                self.augmentation_progress = 0.0
                self.current_operation = None

            # 증강 디렉터리 생성
            augmented_dir = Path(self.config.augmented_data_directory)
            augmented_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info("AugmentationService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize AugmentationService: {str(e)}")
            self._is_initialized = False
            return False

    def cleanup(self) -> None:
        """
        서비스 정리
        """
        try:
            self.logger.info("Cleaning up AugmentationService")

            with self._lock:
                self.source_documents.clear()
                self.augmented_documents.clear()
                self.augmentation_statistics.clear()
                self.processing_errors.clear()

            # 임시 파일 정리
            if hasattr(self.file_handler, "cleanup_temp_files"):
                self.file_handler.cleanup_temp_files()

            self.logger.info("AugmentationService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during AugmentationService cleanup: {str(e)}")

    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 초기화 상태 확인
            if not self.is_initialized():
                self.logger.warning("Service not initialized")
                return False

            # 기본 컴포넌트 확인
            if not hasattr(self, "config") or self.config is None:
                self.logger.warning("Config is None")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    def augment_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        데이터셋 증강 (AugmentationInterface 구현)

        Args:
            dataset: 원본 데이터셋

        Returns:
            List[Dict[str, Any]]: 증강된 데이터셋
        """
        try:
            self.logger.info(f"Starting dataset augmentation with {len(dataset)} items")

            with self._lock:
                self.current_operation = "dataset_augmentation"
                self.augmentation_progress = 0.0

            # 원본 데이터 저장
            augmented_dataset = dataset.copy()

            # 배치 단위로 증강 처리
            batch_size = min(self.batch_size, len(dataset))
            total_batches = (len(dataset) + batch_size - 1) // batch_size

            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(dataset))
                batch_data = dataset[start_idx:end_idx]

                # 배치 증강 처리
                augmented_batch = self._augment_batch(batch_data)
                augmented_dataset.extend(augmented_batch)

                # 진행률 업데이트
                progress = (batch_idx + 1) / total_batches
                self._update_augmentation_progress(progress)

            # 통계 업데이트
            self._update_augmentation_statistics(len(dataset), len(augmented_dataset))

            with self._lock:
                self.current_operation = None
                self.augmentation_progress = 1.0

            # 완료 콜백 실행
            self._execute_completion_callbacks()

            self.logger.info(
                f"Dataset augmentation completed: {len(dataset)} → {len(augmented_dataset)} items"
            )
            return augmented_dataset

        except Exception as e:
            self.logger.error(f"Dataset augmentation failed: {str(e)}")
            with self._lock:
                self.processing_errors.append(str(e))
                self.current_operation = None
            raise AugmentationError(
                message=f"Dataset augmentation failed: {str(e)}",
                augmentation_type="dataset",
                original_exception=e,
            )

    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """
        증강 통계 정보 제공 (AugmentationInterface 구현)

        Returns:
            Dict[str, Any]: 증강 통계 정보
        """
        with self._lock:
            return self.augmentation_statistics.copy()

    def configure_augmentation_rules(self, rules: Dict[str, Any]) -> None:
        """
        증강 규칙 설정 (AugmentationInterface 구현)

        Args:
            rules: 증강 규칙 딕셔너리
        """
        try:
            # 기존 규칙 업데이트
            self.augmentation_rules.update(rules)

            # 규칙 검증
            self._validate_augmentation_rules()

            self.logger.info(
                f"Augmentation rules configured: {len(rules)} rules updated"
            )

        except Exception as e:
            self.logger.error(f"Failed to configure augmentation rules: {str(e)}")
            raise AugmentationError(
                message=f"Failed to configure augmentation rules: {str(e)}",
                augmentation_type="configuration",
                original_exception=e,
            )

    def generate_augmented_dataset(
        self, original_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        증강된 데이터셋 생성

        Args:
            original_data: 원본 데이터 목록

        Returns:
            List[Dict[str, Any]]: 증강된 데이터셋
        """
        try:
            self.logger.info(
                f"Generating augmented dataset from {len(original_data)} items"
            )

            augmented_data = []

            for data_item in original_data:
                try:
                    # 데이터 항목별 증강 처리
                    augmented_items = self._augment_data_item(data_item)
                    augmented_data.extend(augmented_items)

                except Exception as e:
                    self.logger.error(f"Failed to augment data item: {str(e)}")
                    with self._lock:
                        self.processing_errors.append(str(e))

            self.logger.info(f"Generated {len(augmented_data)} augmented items")
            return augmented_data

        except Exception as e:
            self.logger.error(f"Failed to generate augmented dataset: {str(e)}")
            raise AugmentationError(
                message=f"Failed to generate augmented dataset: {str(e)}",
                augmentation_type="generation",
                original_exception=e,
            )

    def apply_geometric_transformations(self, image: np.ndarray) -> List[np.ndarray]:
        """
        기하학적 변환 적용

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 변환된 이미지 목록
        """
        try:
            return self.geometric_transformer.apply_geometric_transformations(image)

        except Exception as e:
            self.logger.error(f"Geometric transformation failed: {str(e)}")
            raise AugmentationError(
                message=f"Geometric transformation failed: {str(e)}",
                augmentation_type="geometric",
                original_exception=e,
            )

    def apply_color_adjustments(self, image: np.ndarray) -> List[np.ndarray]:
        """
        색상 조정 적용

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 색상 조정된 이미지 목록
        """
        try:
            return self.color_adjuster.apply_color_adjustments(image)

        except Exception as e:
            self.logger.error(f"Color adjustment failed: {str(e)}")
            raise AugmentationError(
                message=f"Color adjustment failed: {str(e)}",
                augmentation_type="color",
                original_exception=e,
            )

    def add_noise_variations(self, image: np.ndarray) -> List[np.ndarray]:
        """
        노이즈 변형 추가

        Args:
            image: 입력 이미지 배열

        Returns:
            List[np.ndarray]: 노이즈가 추가된 이미지 목록
        """
        try:
            return self.noise_generator.add_noise_variations(image)

        except Exception as e:
            self.logger.error(f"Noise addition failed: {str(e)}")
            raise AugmentationError(
                message=f"Noise addition failed: {str(e)}",
                augmentation_type="noise",
                original_exception=e,
            )

    def save_augmented_data(
        self, augmented_data: List[Dict[str, Any]], output_path: str
    ) -> bool:
        """
        증강된 데이터 저장

        Args:
            augmented_data: 증강된 데이터 목록
            output_path: 출력 파일 경로

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 출력 디렉터리 생성
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # JSON 형태로 저장
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(augmented_data, f, indent=2, ensure_ascii=False, default=str)

            self.logger.info(f"Augmented data saved to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save augmented data: {str(e)}")
            return False

    def _augment_batch(self, batch_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """배치 데이터 증강"""
        augmented_batch = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_data = {
                executor.submit(self._augment_data_item, data_item): data_item
                for data_item in batch_data
            }

            for future in as_completed(future_to_data):
                data_item = future_to_data[future]
                try:
                    augmented_items = future.result()
                    augmented_batch.extend(augmented_items)
                except Exception as e:
                    self.logger.error(f"Failed to augment batch item: {str(e)}")
                    with self._lock:
                        self.processing_errors.append(str(e))

        return augmented_batch

    def _augment_data_item(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """개별 데이터 항목 증강"""
        augmented_items = []

        try:
            # 데이터 항목 유형 확인
            if "image_path" in data_item:
                # 이미지 데이터 증강
                augmented_items = self._augment_image_data(data_item)
            elif "document_path" in data_item:
                # 문서 데이터 증강
                augmented_items = self._augment_document_data(data_item)
            else:
                # 기본 데이터 증강
                augmented_items = self._augment_generic_data(data_item)

            return augmented_items

        except Exception as e:
            self.logger.error(f"Failed to augment data item: {str(e)}")
            return [data_item]  # 원본 데이터 반환

    def _augment_image_data(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """이미지 데이터 증강"""
        augmented_items = []
        image_path = data_item["image_path"]

        try:
            # 증강 기법 적용
            augmented_paths = self.image_augmenter.augment_image(
                image_path, self.augmentation_rules["enabled_techniques"]
            )

            # 증강된 이미지별 데이터 항목 생성
            for aug_path in augmented_paths:
                aug_item = data_item.copy()
                aug_item["image_path"] = aug_path
                aug_item["is_augmented"] = True
                aug_item["augmentation_type"] = "image"
                aug_item["original_path"] = image_path
                augmented_items.append(aug_item)

            return augmented_items

        except Exception as e:
            self.logger.error(f"Image data augmentation failed: {str(e)}")
            return [data_item]

    def _augment_document_data(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """문서 데이터 증강"""
        augmented_items = []

        try:
            # 문서 내 이미지 페이지 증강
            document_path = data_item["document_path"]

            # 증강 팩터만큼 반복
            for i in range(self.augmentation_factor):
                aug_item = data_item.copy()
                aug_item["is_augmented"] = True
                aug_item["augmentation_type"] = "document"
                aug_item["augmentation_index"] = i
                augmented_items.append(aug_item)

            return augmented_items

        except Exception as e:
            self.logger.error(f"Document data augmentation failed: {str(e)}")
            return [data_item]

    def _augment_generic_data(self, data_item: Dict[str, Any]) -> List[Dict[str, Any]]:
        """일반 데이터 증강"""
        augmented_items = []

        try:
            # 기본 증강 (복제)
            for i in range(self.augmentation_factor):
                aug_item = data_item.copy()
                aug_item["is_augmented"] = True
                aug_item["augmentation_type"] = "generic"
                aug_item["augmentation_index"] = i
                augmented_items.append(aug_item)

            return augmented_items

        except Exception as e:
            self.logger.error(f"Generic data augmentation failed: {str(e)}")
            return [data_item]

    def _validate_augmentation_rules(self) -> None:
        """증강 규칙 검증"""
        required_keys = [
            "enabled_techniques",
            "geometric_enabled",
            "color_enabled",
            "noise_enabled",
        ]

        for key in required_keys:
            if key not in self.augmentation_rules:
                raise ValidationError(f"Missing required augmentation rule: {key}")

        # 증강 기법 검증
        valid_techniques = ["rotation", "scaling", "brightness", "contrast", "noise"]
        for technique in self.augmentation_rules["enabled_techniques"]:
            if technique not in valid_techniques:
                raise ValidationError(f"Invalid augmentation technique: {technique}")

    def _update_augmentation_progress(self, progress: float) -> None:
        """증강 진행률 업데이트"""
        with self._lock:
            self.augmentation_progress = progress

        # 진행률 콜백 실행
        self._execute_progress_callbacks()

    def _update_augmentation_statistics(
        self, original_count: int, augmented_count: int
    ) -> None:
        """증강 통계 업데이트"""
        with self._lock:
            self.augmentation_statistics = {
                "original_data_count": original_count,
                "augmented_data_count": augmented_count,
                "augmentation_ratio": (
                    augmented_count / original_count if original_count > 0 else 0
                ),
                "augmentation_factor": self.augmentation_factor,
                "processing_errors_count": len(self.processing_errors),
                "last_update_time": datetime.now().isoformat(),
                "service_id": self.service_id,
            }

            # 컴포넌트별 통계 추가
            self.augmentation_statistics["image_augmenter_stats"] = (
                self.image_augmenter.get_statistics()
            )

    def _execute_progress_callbacks(self) -> None:
        """진행률 콜백 실행"""
        with self._lock:
            callbacks = self.progress_callbacks.copy()
            progress = self.augmentation_progress

        for callback in callbacks:
            try:
                callback(progress)
            except Exception as e:
                self.logger.error(f"Progress callback execution failed: {str(e)}")

    def _execute_completion_callbacks(self) -> None:
        """완료 콜백 실행"""
        with self._lock:
            callbacks = self.completion_callbacks.copy()
            augmented_docs = self.augmented_documents.copy()

        for callback in callbacks:
            try:
                callback(augmented_docs)
            except Exception as e:
                self.logger.error(f"Completion callback execution failed: {str(e)}")

    def get_augmentation_progress(self) -> Dict[str, Any]:
        """증강 진행 상황 반환"""
        with self._lock:
            return {
                "progress": self.augmentation_progress,
                "current_operation": self.current_operation,
                "source_documents_count": len(self.source_documents),
                "augmented_documents_count": len(self.augmented_documents),
                "processing_errors_count": len(self.processing_errors),
            }

    def register_progress_callback(self, callback: Callable) -> None:
        """진행률 콜백 등록"""
        with self._lock:
            self.progress_callbacks.append(callback)

        self.logger.debug(f"Progress callback registered: {callback.__name__}")

    def register_completion_callback(self, callback: Callable) -> None:
        """완료 콜백 등록"""
        with self._lock:
            self.completion_callbacks.append(callback)

        self.logger.debug(f"Completion callback registered: {callback.__name__}")

    @classmethod
    def create_with_dependencies(cls, container) -> "AugmentationService":
        """
        의존성 컨테이너를 사용한 팩토리 메서드

        Args:
            container: 의존성 컨테이너

        Returns:
            AugmentationService: 생성된 서비스 인스턴스
        """
        return cls(
            config=container.get_service("config"),
            logger=container.get_service("logger"),
        )


# 모듈 수준 유틸리티 함수들
def create_augmentation_service(config: ApplicationConfig) -> AugmentationService:
    """
    데이터 증강 서비스 생성 함수

    Args:
        config: 애플리케이션 설정

    Returns:
        AugmentationService: 생성된 서비스 인스턴스
    """
    logger = get_application_logger("augmentation_service")
    service = AugmentationService(config, logger)

    if not service.initialize():
        raise ProcessingError("Failed to initialize AugmentationService")

    return service


if __name__ == "__main__":
    # 데이터 증강 서비스 테스트
    print("YOKOGAWA OCR 데이터 증강 서비스 테스트")
    print("=" * 50)

    try:
        # 설정 로드
        from config.settings import load_configuration

        config = load_configuration()

        # 서비스 생성
        service = create_augmentation_service(config)

        # 상태 확인
        if service.health_check():
            print("✅ 데이터 증강 서비스 정상 동작")
        else:
            print("❌ 데이터 증강 서비스 상태 이상")

        # 테스트 데이터셋 생성
        test_dataset = [
            {"image_path": "test1.jpg", "label": "document"},
            {"image_path": "test2.jpg", "label": "invoice"},
        ]

        # 데이터셋 증강 테스트
        augmented_dataset = service.augment_dataset(test_dataset)
        print(f"📊 증강 결과: {len(test_dataset)} → {len(augmented_dataset)} 항목")

        # 통계 정보 출력
        statistics = service.get_augmentation_statistics()
        print(f"📈 증강 통계: {statistics}")

        # 정리
        service.cleanup()

    except Exception as e:
        print(f"❌ 데이터 증강 서비스 테스트 실패: {e}")

    print("\n🎯 데이터 증강 서비스 구현이 완료되었습니다!")
