#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 문서 모델 모듈

이 모듈은 문서 데이터를 표현하고 관리하는 모델 클래스들을 정의합니다.
PDF 파일, 이미지 파일 등의 문서 데이터를 추상화하여 처리합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

from core.base_classes import BaseModel
from core.exceptions import (
    ValidationError,
    FileAccessError,
    FileFormatError,
    ProcessingError,
    DataIntegrityError,
)
from config.constants import (
    SUPPORTED_FILE_FORMATS,
    PDF_FILE_EXTENSIONS,
    IMAGE_FILE_EXTENSIONS,
    DEFAULT_IMAGE_RESOLUTION,
    MAX_FILE_SIZE_MB,
    TEXT_MIN_LENGTH,
    TEXT_MAX_LENGTH,
)


class DocumentType(Enum):
    """문서 타입 열거형"""

    PDF = "pdf"
    IMAGE = "image"
    UNKNOWN = "unknown"


class DocumentStatus(Enum):
    """문서 처리 상태 열거형"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


class PageType(Enum):
    """페이지 타입 열거형"""

    COVER = "cover"
    CONTENT = "content"
    APPENDIX = "appendix"
    BLANK = "blank"


# ====================================================================================
# 1. 문서 메타데이터 클래스
# ====================================================================================


@dataclass
class DocumentMetadata:
    """
    문서 메타데이터 클래스

    파일 속성 및 문서 정보를 관리합니다.
    """

    # 파일 정보
    file_path: str
    file_name: str
    file_size: int
    file_extension: str
    file_hash: str

    # 생성/수정 정보
    creation_date: datetime
    modification_date: datetime
    access_date: datetime

    # 문서 속성
    document_title: str = ""
    document_author: str = ""
    document_subject: str = ""
    document_keywords: str = ""
    document_creator: str = ""
    document_producer: str = ""

    # 기술적 정보
    page_count: int = 0
    document_version: str = ""
    encryption_status: bool = False

    # 추가 속성
    custom_properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """초기화 후 검증"""
        self._validate_metadata()

    def _validate_metadata(self) -> None:
        """메타데이터 검증"""
        if not os.path.exists(self.file_path):
            raise FileAccessError(
                message=f"File not found: {self.file_path}", file_path=self.file_path
            )

        if self.file_size <= 0:
            raise ValidationError(
                message=f"Invalid file size: {self.file_size}",
                validation_type="file_size",
            )

        if self.file_extension not in SUPPORTED_FILE_FORMATS:
            raise FileFormatError(
                message=f"Unsupported file format: {self.file_extension}",
                file_path=self.file_path,
                expected_format=", ".join(SUPPORTED_FILE_FORMATS),
                actual_format=self.file_extension,
            )

    def to_dict(self) -> Dict[str, Any]:
        """메타데이터를 딕셔너리로 변환"""
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "file_extension": self.file_extension,
            "file_hash": self.file_hash,
            "creation_date": self.creation_date.isoformat(),
            "modification_date": self.modification_date.isoformat(),
            "access_date": self.access_date.isoformat(),
            "document_title": self.document_title,
            "document_author": self.document_author,
            "document_subject": self.document_subject,
            "document_keywords": self.document_keywords,
            "document_creator": self.document_creator,
            "document_producer": self.document_producer,
            "page_count": self.page_count,
            "document_version": self.document_version,
            "encryption_status": self.encryption_status,
            "custom_properties": self.custom_properties,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentMetadata":
        """딕셔너리에서 메타데이터 인스턴스 생성"""
        return cls(
            file_path=data["file_path"],
            file_name=data["file_name"],
            file_size=data["file_size"],
            file_extension=data["file_extension"],
            file_hash=data["file_hash"],
            creation_date=datetime.fromisoformat(data["creation_date"]),
            modification_date=datetime.fromisoformat(data["modification_date"]),
            access_date=datetime.fromisoformat(data["access_date"]),
            document_title=data.get("document_title", ""),
            document_author=data.get("document_author", ""),
            document_subject=data.get("document_subject", ""),
            document_keywords=data.get("document_keywords", ""),
            document_creator=data.get("document_creator", ""),
            document_producer=data.get("document_producer", ""),
            page_count=data.get("page_count", 0),
            document_version=data.get("document_version", ""),
            encryption_status=data.get("encryption_status", False),
            custom_properties=data.get("custom_properties", {}),
        )

    def update_custom_property(self, key: str, value: Any) -> None:
        """사용자 정의 속성 업데이트"""
        self.custom_properties[key] = value

    def get_custom_property(self, key: str, default: Any = None) -> Any:
        """사용자 정의 속성 조회"""
        return self.custom_properties.get(key, default)


# ====================================================================================
# 2. 페이지 정보 클래스
# ====================================================================================


@dataclass
class PageInfo:
    """
    페이지 정보 클래스

    문서의 개별 페이지 정보를 관리합니다.
    """

    # 기본 정보
    page_number: int
    page_type: PageType = PageType.CONTENT

    # 물리적 속성
    page_width: int = 0
    page_height: int = 0
    page_rotation: int = 0
    page_resolution: int = DEFAULT_IMAGE_RESOLUTION

    # 컨텐츠 정보
    has_text: bool = False
    has_images: bool = False
    has_tables: bool = False
    has_forms: bool = False

    # 텍스트 정보
    text_content: str = ""
    text_length: int = 0
    text_language: str = "ko"

    # 이미지 정보
    image_count: int = 0
    image_paths: List[str] = field(default_factory=list)

    # 품질 정보
    quality_score: float = 0.0
    is_blank: bool = False
    has_noise: bool = False

    # 처리 정보
    processing_time: float = 0.0
    processing_errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """초기화 후 검증"""
        self._validate_page_info()
        self._calculate_derived_properties()

    def _validate_page_info(self) -> None:
        """페이지 정보 검증"""
        if self.page_number < 1:
            raise ValidationError(
                message=f"Invalid page number: {self.page_number}",
                validation_type="page_number",
            )

        if self.page_width < 0 or self.page_height < 0:
            raise ValidationError(
                message=f"Invalid page dimensions: {self.page_width}x{self.page_height}",
                validation_type="page_dimensions",
            )

    def _calculate_derived_properties(self) -> None:
        """파생 속성 계산"""
        self.text_length = len(self.text_content)
        self.image_count = len(self.image_paths)
        self.is_blank = (
            not self.has_text
            and not self.has_images
            and not self.has_tables
            and not self.has_forms
        )

    def to_dict(self) -> Dict[str, Any]:
        """페이지 정보를 딕셔너리로 변환"""
        return {
            "page_number": self.page_number,
            "page_type": self.page_type.value,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "page_rotation": self.page_rotation,
            "page_resolution": self.page_resolution,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "has_tables": self.has_tables,
            "has_forms": self.has_forms,
            "text_content": self.text_content,
            "text_length": self.text_length,
            "text_language": self.text_language,
            "image_count": self.image_count,
            "image_paths": self.image_paths,
            "quality_score": self.quality_score,
            "is_blank": self.is_blank,
            "has_noise": self.has_noise,
            "processing_time": self.processing_time,
            "processing_errors": self.processing_errors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PageInfo":
        """딕셔너리에서 페이지 정보 인스턴스 생성"""
        return cls(
            page_number=data["page_number"],
            page_type=PageType(data.get("page_type", "content")),
            page_width=data.get("page_width", 0),
            page_height=data.get("page_height", 0),
            page_rotation=data.get("page_rotation", 0),
            page_resolution=data.get("page_resolution", DEFAULT_IMAGE_RESOLUTION),
            has_text=data.get("has_text", False),
            has_images=data.get("has_images", False),
            has_tables=data.get("has_tables", False),
            has_forms=data.get("has_forms", False),
            text_content=data.get("text_content", ""),
            text_language=data.get("text_language", "ko"),
            image_paths=data.get("image_paths", []),
            quality_score=data.get("quality_score", 0.0),
            has_noise=data.get("has_noise", False),
            processing_time=data.get("processing_time", 0.0),
            processing_errors=data.get("processing_errors", []),
        )

    def add_image_path(self, image_path: str) -> None:
        """이미지 경로 추가"""
        if image_path not in self.image_paths:
            self.image_paths.append(image_path)
            self.image_count = len(self.image_paths)
            self.has_images = True

    def set_text_content(self, text: str) -> None:
        """텍스트 컨텐츠 설정"""
        self.text_content = text.strip()
        self.text_length = len(self.text_content)
        self.has_text = self.text_length > 0

    def add_processing_error(self, error: str) -> None:
        """처리 오류 추가"""
        self.processing_errors.append(error)

    def get_aspect_ratio(self) -> float:
        """페이지 종횡비 반환"""
        if self.page_height == 0:
            return 0.0
        return self.page_width / self.page_height


# ====================================================================================
# 3. 문서 통계 클래스
# ====================================================================================


@dataclass
class DocumentStatistics:
    """
    문서 통계 정보 클래스

    문서 전체의 통계 정보를 관리합니다.
    """

    # 기본 통계
    total_pages: int = 0
    total_text_length: int = 0
    total_images: int = 0
    total_tables: int = 0
    total_forms: int = 0

    # 페이지 타입별 통계
    cover_pages: int = 0
    content_pages: int = 0
    appendix_pages: int = 0
    blank_pages: int = 0

    # 품질 통계
    average_quality_score: float = 0.0
    min_quality_score: float = 0.0
    max_quality_score: float = 0.0
    pages_with_noise: int = 0

    # 처리 통계
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    pages_with_errors: int = 0
    total_errors: int = 0

    # 언어 통계
    detected_languages: Dict[str, int] = field(default_factory=dict)
    primary_language: str = ""

    def calculate_statistics(self, pages: List[PageInfo]) -> None:
        """페이지 정보 목록에서 통계 계산"""
        if not pages:
            return

        self.total_pages = len(pages)

        # 기본 통계 계산
        self.total_text_length = sum(page.text_length for page in pages)
        self.total_images = sum(page.image_count for page in pages)

        # 페이지 타입별 통계
        type_counts = {}
        for page in pages:
            page_type = page.page_type.value
            type_counts[page_type] = type_counts.get(page_type, 0) + 1

        self.cover_pages = type_counts.get("cover", 0)
        self.content_pages = type_counts.get("content", 0)
        self.appendix_pages = type_counts.get("appendix", 0)
        self.blank_pages = sum(1 for page in pages if page.is_blank)

        # 품질 통계
        quality_scores = [
            page.quality_score for page in pages if page.quality_score > 0
        ]
        if quality_scores:
            self.average_quality_score = sum(quality_scores) / len(quality_scores)
            self.min_quality_score = min(quality_scores)
            self.max_quality_score = max(quality_scores)

        self.pages_with_noise = sum(1 for page in pages if page.has_noise)

        # 처리 통계
        self.total_processing_time = sum(page.processing_time for page in pages)
        if self.total_pages > 0:
            self.average_processing_time = self.total_processing_time / self.total_pages

        self.pages_with_errors = sum(1 for page in pages if page.processing_errors)
        self.total_errors = sum(len(page.processing_errors) for page in pages)

        # 언어 통계
        language_counts = {}
        for page in pages:
            if page.text_language:
                language_counts[page.text_language] = (
                    language_counts.get(page.text_language, 0) + 1
                )

        self.detected_languages = language_counts
        if language_counts:
            self.primary_language = max(language_counts, key=language_counts.get)

    def to_dict(self) -> Dict[str, Any]:
        """통계 정보를 딕셔너리로 변환"""
        return {
            "total_pages": self.total_pages,
            "total_text_length": self.total_text_length,
            "total_images": self.total_images,
            "total_tables": self.total_tables,
            "total_forms": self.total_forms,
            "cover_pages": self.cover_pages,
            "content_pages": self.content_pages,
            "appendix_pages": self.appendix_pages,
            "blank_pages": self.blank_pages,
            "average_quality_score": self.average_quality_score,
            "min_quality_score": self.min_quality_score,
            "max_quality_score": self.max_quality_score,
            "pages_with_noise": self.pages_with_noise,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.average_processing_time,
            "pages_with_errors": self.pages_with_errors,
            "total_errors": self.total_errors,
            "detected_languages": self.detected_languages,
            "primary_language": self.primary_language,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentStatistics":
        """딕셔너리에서 통계 인스턴스 생성"""
        return cls(
            total_pages=data.get("total_pages", 0),
            total_text_length=data.get("total_text_length", 0),
            total_images=data.get("total_images", 0),
            total_tables=data.get("total_tables", 0),
            total_forms=data.get("total_forms", 0),
            cover_pages=data.get("cover_pages", 0),
            content_pages=data.get("content_pages", 0),
            appendix_pages=data.get("appendix_pages", 0),
            blank_pages=data.get("blank_pages", 0),
            average_quality_score=data.get("average_quality_score", 0.0),
            min_quality_score=data.get("min_quality_score", 0.0),
            max_quality_score=data.get("max_quality_score", 0.0),
            pages_with_noise=data.get("pages_with_noise", 0),
            total_processing_time=data.get("total_processing_time", 0.0),
            average_processing_time=data.get("average_processing_time", 0.0),
            pages_with_errors=data.get("pages_with_errors", 0),
            total_errors=data.get("total_errors", 0),
            detected_languages=data.get("detected_languages", {}),
            primary_language=data.get("primary_language", ""),
        )


# ====================================================================================
# 4. 메인 문서 모델 클래스
# ====================================================================================


class DocumentModel(BaseModel):
    """
    문서 모델 클래스

    PDF 파일이나 이미지 파일을 추상화한 문서 모델입니다.
    BaseModel을 상속받아 표준 모델 인터페이스를 구현합니다.
    """

    def __init__(
        self,
        file_path: str,
        metadata: Optional[DocumentMetadata] = None,
        document_type: Optional[DocumentType] = None,
    ):
        """
        DocumentModel 초기화

        Args:
            file_path: 문서 파일 경로
            metadata: 문서 메타데이터 (없으면 자동 생성)
            document_type: 문서 타입 (없으면 자동 감지)
        """
        super().__init__()

        # 기본 정보
        self.file_path = file_path
        self.document_id = str(uuid.uuid4())
        self.document_type = document_type or self._detect_document_type(file_path)
        self.document_status = DocumentStatus.PENDING

        # 메타데이터
        self.metadata = metadata or self._extract_metadata(file_path)

        # 페이지 정보
        self.pages: List[PageInfo] = []
        self.current_page_index = 0

        # 통계 정보
        self.statistics = DocumentStatistics()

        # 어노테이션 관련
        self.annotations: List[Any] = []  # AnnotationModel 순환 의존성 방지
        self.annotation_count = 0
        self.annotation_progress = 0.0

        # 처리 정보
        self.processing_history: List[Dict[str, Any]] = []
        self.last_processed_time: Optional[datetime] = None
        self.processing_errors: List[str] = []

        # 검증 정보
        self.is_validated = False
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []

        # 초기 검증
        self._validate_initial_state()

    def _detect_document_type(self, file_path: str) -> DocumentType:
        """파일 경로에서 문서 타입 감지"""
        file_extension = Path(file_path).suffix.lower()

        if file_extension in PDF_FILE_EXTENSIONS:
            return DocumentType.PDF
        elif file_extension in IMAGE_FILE_EXTENSIONS:
            return DocumentType.IMAGE
        else:
            return DocumentType.UNKNOWN

    def _extract_metadata(self, file_path: str) -> DocumentMetadata:
        """파일에서 메타데이터 추출"""
        try:
            file_path = os.path.abspath(file_path)
            file_stat = os.stat(file_path)

            # 파일 해시 계산
            file_hash = self._calculate_file_hash(file_path)

            # 기본 메타데이터 생성
            metadata = DocumentMetadata(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=file_stat.st_size,
                file_extension=Path(file_path).suffix.lower(),
                file_hash=file_hash,
                creation_date=datetime.fromtimestamp(file_stat.st_ctime),
                modification_date=datetime.fromtimestamp(file_stat.st_mtime),
                access_date=datetime.fromtimestamp(file_stat.st_atime),
            )

            # 문서 타입별 추가 정보 추출
            if self.document_type == DocumentType.PDF:
                self._extract_pdf_metadata(metadata)
            elif self.document_type == DocumentType.IMAGE:
                self._extract_image_metadata(metadata)

            return metadata

        except Exception as e:
            raise ProcessingError(
                message=f"Failed to extract metadata from {file_path}: {str(e)}",
                processor_id=self.model_id,
                processing_stage="metadata_extraction",
                original_exception=e,
            )

    def _calculate_file_hash(self, file_path: str) -> str:
        """파일 해시 계산"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            raise ProcessingError(
                message=f"Failed to calculate file hash: {str(e)}", original_exception=e
            )

    def _extract_pdf_metadata(self, metadata: DocumentMetadata) -> None:
        """PDF 메타데이터 추출"""
        try:
            # PDF 라이브러리 import (지연 import로 의존성 최소화)
            import fitz  # PyMuPDF

            pdf_document = fitz.open(metadata.file_path)

            # 페이지 수 설정
            metadata.page_count = len(pdf_document)

            # 문서 정보 추출
            pdf_metadata = pdf_document.metadata
            metadata.document_title = pdf_metadata.get("title", "")
            metadata.document_author = pdf_metadata.get("author", "")
            metadata.document_subject = pdf_metadata.get("subject", "")
            metadata.document_keywords = pdf_metadata.get("keywords", "")
            metadata.document_creator = pdf_metadata.get("creator", "")
            metadata.document_producer = pdf_metadata.get("producer", "")

            # 암호화 상태 확인
            metadata.encryption_status = pdf_document.needs_pass

            pdf_document.close()

        except ImportError:
            self.add_validation_error("PyMuPDF not available for PDF processing")
        except Exception as e:
            self.add_validation_error(f"PDF metadata extraction failed: {str(e)}")

    def _extract_image_metadata(self, metadata: DocumentMetadata) -> None:
        """이미지 메타데이터 추출"""
        try:
            # PIL 라이브러리 import (지연 import로 의존성 최소화)
            from PIL import Image

            with Image.open(metadata.file_path) as img:
                # 이미지 정보
                metadata.custom_properties.update(
                    {
                        "image_format": img.format,
                        "image_mode": img.mode,
                        "image_size": img.size,
                        "image_has_transparency": img.mode in ("RGBA", "LA")
                        or "transparency" in img.info,
                    }
                )

                # EXIF 데이터 추출
                if hasattr(img, "_getexif") and img._getexif():
                    exif_data = img._getexif()
                    metadata.custom_properties["exif_data"] = exif_data

                # 이미지는 단일 페이지로 처리
                metadata.page_count = 1

        except ImportError:
            self.add_validation_error("PIL not available for image processing")
        except Exception as e:
            self.add_validation_error(f"Image metadata extraction failed: {str(e)}")

    def _validate_initial_state(self) -> None:
        """초기 상태 검증"""
        if not os.path.exists(self.file_path):
            raise FileAccessError(
                message=f"File not found: {self.file_path}", file_path=self.file_path
            )

        if self.document_type == DocumentType.UNKNOWN:
            self.add_validation_error(
                f"Unknown document type for file: {self.file_path}"
            )

        # 파일 크기 검증
        file_size_mb = self.metadata.file_size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise ValidationError(
                message=f"File size ({file_size_mb:.2f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)",
                validation_type="file_size",
            )

    # BaseModel 추상 메서드 구현
    def to_dict(self) -> Dict[str, Any]:
        """
        문서 모델을 딕셔너리로 변환

        Returns:
            Dict[str, Any]: 문서 모델 딕셔너리
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "document_status": self.document_status.value,
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "statistics": self.statistics.to_dict(),
            "annotation_count": self.annotation_count,
            "annotation_progress": self.annotation_progress,
            "processing_history": self.processing_history,
            "last_processed_time": (
                self.last_processed_time.isoformat()
                if self.last_processed_time
                else None
            ),
            "processing_errors": self.processing_errors,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentModel":
        """
        딕셔너리에서 문서 모델 인스턴스 생성

        Args:
            data: 문서 모델 데이터 딕셔너리

        Returns:
            DocumentModel: 문서 모델 인스턴스
        """
        # 메타데이터 복원
        metadata = DocumentMetadata.from_dict(data["metadata"])

        # 문서 타입 복원
        document_type = DocumentType(data["document_type"])

        # 인스턴스 생성
        instance = cls(
            file_path=data["file_path"], metadata=metadata, document_type=document_type
        )

        # 추가 속성 복원
        instance.document_id = data["document_id"]
        instance.document_status = DocumentStatus(data["document_status"])

        # 페이지 정보 복원
        instance.pages = [PageInfo.from_dict(page_data) for page_data in data["pages"]]

        # 통계 정보 복원
        instance.statistics = DocumentStatistics.from_dict(data["statistics"])

        # 어노테이션 정보 복원
        instance.annotation_count = data.get("annotation_count", 0)
        instance.annotation_progress = data.get("annotation_progress", 0.0)

        # 처리 정보 복원
        instance.processing_history = data.get("processing_history", [])
        if data.get("last_processed_time"):
            instance.last_processed_time = datetime.fromisoformat(
                data["last_processed_time"]
            )
        instance.processing_errors = data.get("processing_errors", [])

        # 검증 정보 복원
        instance.is_validated = data.get("is_validated", False)
        instance.validation_errors = data.get("validation_errors", [])
        instance.validation_warnings = data.get("validation_warnings", [])

        return instance

    def validate(self) -> bool:
        """
        문서 모델 데이터 유효성 검증

        Returns:
            bool: 유효성 검증 결과
        """
        self.clear_validation_errors()

        try:
            # 파일 존재 여부 확인
            if not os.path.exists(self.file_path):
                self.add_validation_error(f"File not found: {self.file_path}")

            # 메타데이터 검증
            if not self.metadata:
                self.add_validation_error("Document metadata is missing")

            # 문서 타입 검증
            if self.document_type == DocumentType.UNKNOWN:
                self.add_validation_error("Document type is unknown")

            # 페이지 정보 검증
            if self.metadata.page_count > 0 and not self.pages:
                self.add_validation_error("Page information is missing")

            # 페이지 번호 연속성 검증
            if self.pages:
                expected_page_numbers = list(range(1, len(self.pages) + 1))
                actual_page_numbers = [page.page_number for page in self.pages]
                if actual_page_numbers != expected_page_numbers:
                    self.add_validation_error("Page numbers are not consecutive")

            # 어노테이션 일관성 검증
            if self.annotation_count != len(self.annotations):
                self.add_validation_error("Annotation count mismatch")

            # 진행률 검증
            if not (0.0 <= self.annotation_progress <= 1.0):
                self.add_validation_error("Invalid annotation progress value")

            # 검증 완료 표시
            self.is_validated = len(self.validation_errors) == 0

            return self.is_validated

        except Exception as e:
            self.add_validation_error(f"Validation failed: {str(e)}")
            return False

    # 추가 메서드들
    @classmethod
    def from_file_path(cls, file_path: str) -> "DocumentModel":
        """
        파일 경로에서 문서 모델 생성

        Args:
            file_path: 문서 파일 경로

        Returns:
            DocumentModel: 문서 모델 인스턴스
        """
        return cls(file_path=file_path)

    def validate_document_format(self) -> bool:
        """
        문서 형식 유효성 검증

        Returns:
            bool: 형식 유효성 검증 결과
        """
        try:
            # 파일 확장자 검증
            if self.metadata.file_extension not in SUPPORTED_FILE_FORMATS:
                return False

            # 파일 크기 검증
            file_size_mb = self.metadata.file_size / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                return False

            # 문서 타입별 형식 검증
            if self.document_type == DocumentType.PDF:
                return self._validate_pdf_format()
            elif self.document_type == DocumentType.IMAGE:
                return self._validate_image_format()

            return True

        except Exception as e:
            self.add_validation_error(f"Format validation failed: {str(e)}")
            return False

    def _validate_pdf_format(self) -> bool:
        """PDF 형식 검증"""
        try:
            import fitz

            pdf_document = fitz.open(self.file_path)
            is_valid = len(pdf_document) > 0
            pdf_document.close()
            return is_valid
        except Exception:
            return False

    def _validate_image_format(self) -> bool:
        """이미지 형식 검증"""
        try:
            from PIL import Image

            with Image.open(self.file_path) as img:
                img.verify()
            return True
        except Exception:
            return False

    def extract_text_content(self) -> str:
        """
        문서에서 텍스트 컨텐츠 추출

        Returns:
            str: 추출된 텍스트 내용
        """
        if not self.pages:
            return ""

        text_content = []
        for page in self.pages:
            if page.text_content:
                text_content.append(page.text_content)

        return "\n\n".join(text_content)

    def get_page_count(self) -> int:
        """
        페이지 수 반환

        Returns:
            int: 페이지 수
        """
        return self.metadata.page_count

    def add_page(self, page_info: PageInfo) -> None:
        """
        페이지 정보 추가

        Args:
            page_info: 추가할 페이지 정보
        """
        self.pages.append(page_info)
        self.metadata.page_count = len(self.pages)
        self.statistics.calculate_statistics(self.pages)
        self._updated_at = datetime.now()
        self._version += 1

    def get_page(self, page_number: int) -> Optional[PageInfo]:
        """
        특정 페이지 정보 조회

        Args:
            page_number: 페이지 번호 (1부터 시작)

        Returns:
            Optional[PageInfo]: 페이지 정보 (없으면 None)
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def add_annotation(self, annotation: Any) -> None:
        """
        어노테이션 추가

        Args:
            annotation: 추가할 어노테이션 객체
        """
        self.annotations.append(annotation)
        self.annotation_count = len(self.annotations)
        self._calculate_annotation_progress()
        self._updated_at = datetime.now()
        self._version += 1

    def _calculate_annotation_progress(self) -> None:
        """어노테이션 진행률 계산"""
        if not self.pages:
            self.annotation_progress = 0.0
            return

        # 간단한 진행률 계산 (페이지당 어노테이션 존재 여부 기준)
        annotated_pages = sum(
            1 for page in self.pages if page.has_text or page.has_images
        )
        if annotated_pages == 0:
            self.annotation_progress = 0.0
        else:
            self.annotation_progress = min(self.annotation_count / annotated_pages, 1.0)

    def add_processing_history(self, operation: str, result: str, **kwargs) -> None:
        """
        처리 이력 추가

        Args:
            operation: 수행된 작업
            result: 작업 결과
            **kwargs: 추가 정보
        """
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "result": result,
            **kwargs,
        }
        self.processing_history.append(history_entry)
        self.last_processed_time = datetime.now()
        self._updated_at = datetime.now()
        self._version += 1

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        처리 요약 정보 반환

        Returns:
            Dict[str, Any]: 처리 요약 정보
        """
        return {
            "document_id": self.document_id,
            "file_path": self.file_path,
            "document_type": self.document_type.value,
            "document_status": self.document_status.value,
            "page_count": self.get_page_count(),
            "annotation_count": self.annotation_count,
            "annotation_progress": self.annotation_progress,
            "processing_operations": len(self.processing_history),
            "processing_errors": len(self.processing_errors),
            "validation_status": self.is_validated,
            "last_processed": (
                self.last_processed_time.isoformat()
                if self.last_processed_time
                else None
            ),
            "creation_time": self.created_at.isoformat(),
            "last_updated": self.updated_at.isoformat(),
        }

    def clone(self) -> "DocumentModel":
        """
        문서 모델 복제

        Returns:
            DocumentModel: 복제된 문서 모델
        """
        cloned_dict = self.to_dict()
        cloned_dict["document_id"] = str(uuid.uuid4())  # 새로운 ID 생성
        return self.from_dict(cloned_dict)

    def __str__(self) -> str:
        """문자열 표현"""
        return f"DocumentModel(id={self.document_id}, file={self.metadata.file_name}, pages={self.get_page_count()})"

    def __repr__(self) -> str:
        """개발자용 표현"""
        return (
            f"DocumentModel("
            f"document_id='{self.document_id}', "
            f"file_path='{self.file_path}', "
            f"document_type={self.document_type.name}, "
            f"page_count={self.get_page_count()}, "
            f"annotation_count={self.annotation_count}"
            f")"
        )


# ====================================================================================
# 5. 유틸리티 함수들
# ====================================================================================


def create_document_from_file(file_path: str) -> DocumentModel:
    """
    파일에서 문서 모델 생성

    Args:
        file_path: 파일 경로

    Returns:
        DocumentModel: 생성된 문서 모델
    """
    return DocumentModel.from_file_path(file_path)


def validate_document_batch(documents: List[DocumentModel]) -> Dict[str, Any]:
    """
    문서 배치 검증

    Args:
        documents: 검증할 문서 목록

    Returns:
        Dict[str, Any]: 배치 검증 결과
    """
    results = {
        "total_documents": len(documents),
        "valid_documents": 0,
        "invalid_documents": 0,
        "validation_errors": [],
        "validation_warnings": [],
    }

    for doc in documents:
        is_valid = doc.validate()
        if is_valid:
            results["valid_documents"] += 1
        else:
            results["invalid_documents"] += 1
            results["validation_errors"].extend(doc.validation_errors)
            results["validation_warnings"].extend(doc.validation_warnings)

    return results


def merge_document_statistics(documents: List[DocumentModel]) -> DocumentStatistics:
    """
    다중 문서 통계 정보 병합

    Args:
        documents: 문서 목록

    Returns:
        DocumentStatistics: 병합된 통계 정보
    """
    merged_stats = DocumentStatistics()

    # 모든 페이지 정보 수집
    all_pages = []
    for doc in documents:
        all_pages.extend(doc.pages)

    # 통계 계산
    merged_stats.calculate_statistics(all_pages)

    return merged_stats


if __name__ == "__main__":
    # 문서 모델 테스트
    print("YOKOGAWA OCR 문서 모델 테스트")
    print("=" * 50)

    try:
        # 테스트용 더미 파일 생성 (실제 사용 시에는 실제 파일 경로 사용)
        test_file_path = "test_document.pdf"

        # 더미 파일이 없으면 테스트 스킵
        if not os.path.exists(test_file_path):
            print("⚠️  테스트 파일이 없습니다. 실제 파일 경로를 사용하여 테스트하세요.")
        else:
            # 문서 모델 생성
            document = DocumentModel.from_file_path(test_file_path)
            print(f"✅ 문서 모델 생성 완료: {document.document_id}")

            # 검증 수행
            is_valid = document.validate()
            print(f"✅ 문서 검증 결과: {'통과' if is_valid else '실패'}")

            # 요약 정보 출력
            summary = document.get_processing_summary()
            print(f"📊 문서 요약: {summary}")

    except Exception as e:
        print(f"❌ 문서 모델 테스트 실패: {e}")

    print("\n🎯 문서 모델 구현이 완료되었습니다!")
