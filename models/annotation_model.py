#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 어노테이션 모델 모듈

이 모듈은 어노테이션 데이터를 표현하고 관리하는 모델 클래스들을 정의합니다.
바운딩 박스, 필드 어노테이션, 문서 어노테이션 등을 추상화하여 처리합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

from core.base_classes import BaseModel
from core.exceptions import (
    ValidationError,
    ProcessingError,
    DataIntegrityError,
    ApplicationError,
)
from config.constants import (
    ANNOTATION_FIELD_TYPES,
    ANNOTATION_MIN_TEXT_LENGTH,
    ANNOTATION_MAX_TEXT_LENGTH,
    ANNOTATION_MIN_BOUNDING_BOX_SIZE,
    ANNOTATION_MAX_BOUNDING_BOX_SIZE,
    ANNOTATION_QUALITY_THRESHOLD,
    ANNOTATION_CONFIDENCE_THRESHOLD,
    BOUNDING_BOX_MIN_WIDTH,
    BOUNDING_BOX_MIN_HEIGHT,
    BOUNDING_BOX_MAX_ASPECT_RATIO,
    BOUNDING_BOX_MIN_ASPECT_RATIO,
)


class AnnotationType(Enum):
    """어노테이션 타입 열거형"""

    TEXT = "text"
    NUMBER = "number"
    DATE = "date"
    CHECKBOX = "checkbox"
    DROPDOWN = "dropdown"
    SIGNATURE = "signature"
    TABLE = "table"
    IMAGE = "image"


class AnnotationStatus(Enum):
    """어노테이션 상태 열거형"""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VALIDATED = "validated"
    REJECTED = "rejected"


class CoordinateSystem(Enum):
    """좌표계 열거형"""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
    NORMALIZED = "normalized"


# ====================================================================================
# 1. 바운딩 박스 클래스
# ====================================================================================


@dataclass
class BoundingBox:
    """
    바운딩 박스 클래스

    어노테이션의 위치 및 크기 정보를 관리합니다.
    """

    # 좌표 정보
    x: int
    y: int
    width: int
    height: int

    # 좌표계 정보
    coordinate_system: CoordinateSystem = CoordinateSystem.ABSOLUTE
    page_number: int = 1

    # 회전 정보
    rotation_angle: float = 0.0

    # 품질 정보
    confidence_score: float = 1.0
    detection_method: str = "manual"

    def __post_init__(self) -> None:
        """초기화 후 검증"""
        self._validate_coordinates()
        self._calculate_derived_properties()

    def _validate_coordinates(self) -> None:
        """좌표 유효성 검증"""
        if self.width < BOUNDING_BOX_MIN_WIDTH:
            raise ValidationError(
                message=f"Bounding box width ({self.width}) is too small. Minimum: {BOUNDING_BOX_MIN_WIDTH}",
                validation_type="bounding_box_width",
            )

        if self.height < BOUNDING_BOX_MIN_HEIGHT:
            raise ValidationError(
                message=f"Bounding box height ({self.height}) is too small. Minimum: {BOUNDING_BOX_MIN_HEIGHT}",
                validation_type="bounding_box_height",
            )

        if self.x < 0 or self.y < 0:
            raise ValidationError(
                message=f"Bounding box coordinates cannot be negative: ({self.x}, {self.y})",
                validation_type="bounding_box_coordinates",
            )

        # 종횡비 검증
        aspect_ratio = self.width / self.height
        if (
            aspect_ratio > BOUNDING_BOX_MAX_ASPECT_RATIO
            or aspect_ratio < BOUNDING_BOX_MIN_ASPECT_RATIO
        ):
            raise ValidationError(
                message=f"Bounding box aspect ratio ({aspect_ratio:.2f}) is out of valid range",
                validation_type="bounding_box_aspect_ratio",
            )

    def _calculate_derived_properties(self) -> None:
        """파생 속성 계산"""
        self.center_x = self.x + self.width // 2
        self.center_y = self.y + self.height // 2
        self.area = self.width * self.height
        self.perimeter = 2 * (self.width + self.height)
        self.aspect_ratio = self.width / self.height

    def to_dict(self) -> Dict[str, Any]:
        """바운딩 박스를 딕셔너리로 변환"""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "coordinate_system": self.coordinate_system.value,
            "page_number": self.page_number,
            "rotation_angle": self.rotation_angle,
            "confidence_score": self.confidence_score,
            "detection_method": self.detection_method,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "area": self.area,
            "perimeter": self.perimeter,
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BoundingBox":
        """딕셔너리에서 바운딩 박스 인스턴스 생성"""
        return cls(
            x=data["x"],
            y=data["y"],
            width=data["width"],
            height=data["height"],
            coordinate_system=CoordinateSystem(
                data.get("coordinate_system", "absolute")
            ),
            page_number=data.get("page_number", 1),
            rotation_angle=data.get("rotation_angle", 0.0),
            confidence_score=data.get("confidence_score", 1.0),
            detection_method=data.get("detection_method", "manual"),
        )

    def get_corners(self) -> List[Tuple[int, int]]:
        """바운딩 박스 모서리 좌표 반환"""
        return [
            (self.x, self.y),  # 좌상단
            (self.x + self.width, self.y),  # 우상단
            (self.x + self.width, self.y + self.height),  # 우하단
            (self.x, self.y + self.height),  # 좌하단
        ]

    def contains_point(self, point_x: int, point_y: int) -> bool:
        """점이 바운딩 박스 내부에 있는지 확인"""
        return (
            self.x <= point_x <= self.x + self.width
            and self.y <= point_y <= self.y + self.height
        )

    def intersects_with(self, other: "BoundingBox") -> bool:
        """다른 바운딩 박스와 교차하는지 확인"""
        return not (
            self.x + self.width < other.x
            or other.x + other.width < self.x
            or self.y + self.height < other.y
            or other.y + other.height < self.y
        )

    def intersection_area(self, other: "BoundingBox") -> int:
        """다른 바운딩 박스와의 교차 면적 계산"""
        if not self.intersects_with(other):
            return 0

        x_overlap = max(
            0, min(self.x + self.width, other.x + other.width) - max(self.x, other.x)
        )
        y_overlap = max(
            0, min(self.y + self.height, other.y + other.height) - max(self.y, other.y)
        )

        return x_overlap * y_overlap

    def union_area(self, other: "BoundingBox") -> int:
        """다른 바운딩 박스와의 합집합 면적 계산"""
        intersection = self.intersection_area(other)
        return self.area + other.area - intersection

    def iou(self, other: "BoundingBox") -> float:
        """다른 바운딩 박스와의 IoU (Intersection over Union) 계산"""
        intersection = self.intersection_area(other)
        union = self.union_area(other)

        if union == 0:
            return 0.0

        return intersection / union

    def scale(self, scale_x: float, scale_y: float) -> "BoundingBox":
        """바운딩 박스 크기 조정"""
        return BoundingBox(
            x=int(self.x * scale_x),
            y=int(self.y * scale_y),
            width=int(self.width * scale_x),
            height=int(self.height * scale_y),
            coordinate_system=self.coordinate_system,
            page_number=self.page_number,
            rotation_angle=self.rotation_angle,
            confidence_score=self.confidence_score,
            detection_method=self.detection_method,
        )

    def normalize(self, image_width: int, image_height: int) -> "BoundingBox":
        """바운딩 박스를 정규화 좌표계로 변환"""
        return BoundingBox(
            x=int(self.x / image_width * 1000),  # 0-1000 스케일
            y=int(self.y / image_height * 1000),
            width=int(self.width / image_width * 1000),
            height=int(self.height / image_height * 1000),
            coordinate_system=CoordinateSystem.NORMALIZED,
            page_number=self.page_number,
            rotation_angle=self.rotation_angle,
            confidence_score=self.confidence_score,
            detection_method=self.detection_method,
        )

    def __str__(self) -> str:
        """문자열 표현"""
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

    def __repr__(self) -> str:
        """개발자용 표현"""
        return (
            f"BoundingBox("
            f"x={self.x}, y={self.y}, "
            f"width={self.width}, height={self.height}, "
            f"page={self.page_number}, "
            f"confidence={self.confidence_score:.2f}"
            f")"
        )


# ====================================================================================
# 2. 필드 어노테이션 클래스
# ====================================================================================


@dataclass
class FieldAnnotation:
    """
    필드 어노테이션 클래스

    개별 필드의 어노테이션 정보를 관리합니다.
    """

    # 기본 정보
    field_id: str
    field_name: str
    field_type: AnnotationType

    # 위치 정보
    bounding_box: BoundingBox

    # 값 정보
    text_value: str = ""
    original_text: str = ""
    normalized_value: Any = None

    # 품질 정보
    confidence_score: float = 1.0
    quality_score: float = 1.0

    # 검증 정보
    is_validated: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # 관계 정보
    parent_field_id: Optional[str] = None
    child_field_ids: List[str] = field(default_factory=list)

    # 메타데이터
    created_by: str = "system"
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        """초기화 후 검증"""
        if not self.field_id:
            self.field_id = str(uuid.uuid4())

        self._validate_field_annotation()
        self._normalize_value()

    def _validate_field_annotation(self) -> None:
        """필드 어노테이션 검증"""
        if not self.field_name:
            raise ValidationError(
                message="Field name cannot be empty", validation_type="field_name"
            )

        if self.field_type.value not in ANNOTATION_FIELD_TYPES:
            raise ValidationError(
                message=f"Invalid field type: {self.field_type.value}",
                validation_type="field_type",
            )

        # 텍스트 길이 검증
        if self.text_value and len(self.text_value) > ANNOTATION_MAX_TEXT_LENGTH:
            raise ValidationError(
                message=f"Text value too long: {len(self.text_value)} > {ANNOTATION_MAX_TEXT_LENGTH}",
                validation_type="text_length",
            )

        # 신뢰도 검증
        if not (0.0 <= self.confidence_score <= 1.0):
            raise ValidationError(
                message=f"Invalid confidence score: {self.confidence_score}",
                validation_type="confidence_score",
            )

    def _normalize_value(self) -> None:
        """값 정규화"""
        if not self.text_value:
            self.normalized_value = None
            return

        try:
            if self.field_type == AnnotationType.NUMBER:
                # 숫자 정규화
                self.normalized_value = float(self.text_value.replace(",", ""))
            elif self.field_type == AnnotationType.DATE:
                # 날짜 정규화
                self.normalized_value = self._parse_date(self.text_value)
            elif self.field_type == AnnotationType.CHECKBOX:
                # 체크박스 정규화
                self.normalized_value = self.text_value.lower() in [
                    "true",
                    "1",
                    "yes",
                    "on",
                    "checked",
                ]
            else:
                # 기본 텍스트 정규화
                self.normalized_value = self.text_value.strip()
        except Exception as e:
            self.add_validation_warning(f"Value normalization failed: {str(e)}")
            self.normalized_value = self.text_value

    def _parse_date(self, date_string: str) -> Optional[datetime]:
        """날짜 문자열 파싱"""
        date_formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
            "%d/%m/%Y",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_string.strip(), fmt)
            except ValueError:
                continue

        return None

    def add_validation_error(self, error: str) -> None:
        """검증 오류 추가"""
        self.validation_errors.append(error)
        self.is_validated = False

    def add_validation_warning(self, warning: str) -> None:
        """검증 경고 추가"""
        self.validation_warnings.append(warning)

    def clear_validation_issues(self) -> None:
        """검증 이슈 초기화"""
        self.validation_errors.clear()
        self.validation_warnings.clear()

    def validate_field_value(self) -> bool:
        """필드 값 유효성 검증"""
        self.clear_validation_issues()

        try:
            # 필수 값 검증
            if not self.text_value and self.field_type != AnnotationType.CHECKBOX:
                self.add_validation_error("Field value is required")
                return False

            # 타입별 검증
            if self.field_type == AnnotationType.NUMBER:
                if not self._validate_number_field():
                    return False
            elif self.field_type == AnnotationType.DATE:
                if not self._validate_date_field():
                    return False
            elif self.field_type == AnnotationType.TEXT:
                if not self._validate_text_field():
                    return False

            # 품질 검증
            if self.confidence_score < ANNOTATION_CONFIDENCE_THRESHOLD:
                self.add_validation_warning(
                    f"Low confidence score: {self.confidence_score}"
                )

            if self.quality_score < ANNOTATION_QUALITY_THRESHOLD:
                self.add_validation_warning(f"Low quality score: {self.quality_score}")

            self.is_validated = len(self.validation_errors) == 0
            return self.is_validated

        except Exception as e:
            self.add_validation_error(f"Validation failed: {str(e)}")
            return False

    def _validate_number_field(self) -> bool:
        """숫자 필드 검증"""
        try:
            float(self.text_value.replace(",", ""))
            return True
        except ValueError:
            self.add_validation_error(f"Invalid number format: {self.text_value}")
            return False

    def _validate_date_field(self) -> bool:
        """날짜 필드 검증"""
        if self._parse_date(self.text_value) is None:
            self.add_validation_error(f"Invalid date format: {self.text_value}")
            return False
        return True

    def _validate_text_field(self) -> bool:
        """텍스트 필드 검증"""
        if len(self.text_value) < ANNOTATION_MIN_TEXT_LENGTH:
            self.add_validation_error(
                f"Text too short: {len(self.text_value)} < {ANNOTATION_MIN_TEXT_LENGTH}"
            )
            return False
        return True

    def update_value(self, new_value: str) -> None:
        """필드 값 업데이트"""
        self.text_value = new_value
        self.modified_at = datetime.now()
        self._normalize_value()
        self.is_validated = False

    def set_bounding_box(self, bounding_box: BoundingBox) -> None:
        """바운딩 박스 설정"""
        self.bounding_box = bounding_box
        self.modified_at = datetime.now()

    def add_child_field(self, child_field_id: str) -> None:
        """자식 필드 추가"""
        if child_field_id not in self.child_field_ids:
            self.child_field_ids.append(child_field_id)
            self.modified_at = datetime.now()

    def remove_child_field(self, child_field_id: str) -> None:
        """자식 필드 제거"""
        if child_field_id in self.child_field_ids:
            self.child_field_ids.remove(child_field_id)
            self.modified_at = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """필드 어노테이션을 딕셔너리로 변환"""
        return {
            "field_id": self.field_id,
            "field_name": self.field_name,
            "field_type": self.field_type.value,
            "bounding_box": self.bounding_box.to_dict(),
            "text_value": self.text_value,
            "original_text": self.original_text,
            "normalized_value": self.normalized_value,
            "confidence_score": self.confidence_score,
            "quality_score": self.quality_score,
            "is_validated": self.is_validated,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
            "parent_field_id": self.parent_field_id,
            "child_field_ids": self.child_field_ids,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FieldAnnotation":
        """딕셔너리에서 필드 어노테이션 인스턴스 생성"""
        return cls(
            field_id=data["field_id"],
            field_name=data["field_name"],
            field_type=AnnotationType(data["field_type"]),
            bounding_box=BoundingBox.from_dict(data["bounding_box"]),
            text_value=data.get("text_value", ""),
            original_text=data.get("original_text", ""),
            normalized_value=data.get("normalized_value"),
            confidence_score=data.get("confidence_score", 1.0),
            quality_score=data.get("quality_score", 1.0),
            is_validated=data.get("is_validated", False),
            validation_errors=data.get("validation_errors", []),
            validation_warnings=data.get("validation_warnings", []),
            parent_field_id=data.get("parent_field_id"),
            child_field_ids=data.get("child_field_ids", []),
            created_by=data.get("created_by", "system"),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if data.get("created_at")
                else datetime.now()
            ),
            modified_at=(
                datetime.fromisoformat(data["modified_at"])
                if data.get("modified_at")
                else datetime.now()
            ),
        )


# ====================================================================================
# 3. 메인 어노테이션 모델 클래스
# ====================================================================================


class AnnotationModel(BaseModel):
    """
    어노테이션 모델 클래스

    문서의 어노테이션 정보를 관리하는 메인 모델입니다.
    BaseModel을 상속받아 표준 모델 인터페이스를 구현합니다.
    """

    def __init__(
        self,
        document_id: str,
        page_number: int = 1,
        annotation_type: AnnotationType = AnnotationType.TEXT,
    ):
        """
        AnnotationModel 초기화

        Args:
            document_id: 문서 ID
            page_number: 페이지 번호
            annotation_type: 어노테이션 타입
        """
        super().__init__()

        # 기본 정보
        self.annotation_id = str(uuid.uuid4())
        self.document_id = document_id
        self.page_number = page_number
        self.annotation_type = annotation_type
        self.annotation_status = AnnotationStatus.PENDING

        # 필드 어노테이션
        self.field_annotations: Dict[str, FieldAnnotation] = {}
        self.field_count = 0

        # 전체 어노테이션 품질 정보
        self.overall_confidence: float = 0.0
        self.overall_quality: float = 0.0
        self.completion_percentage: float = 0.0

        # 세션 정보
        self.session_id: Optional[str] = None
        self.annotator_id: Optional[str] = None
        self.annotation_time: Optional[datetime] = None

        # 검증 정보
        self.validation_results: Dict[str, Any] = {}
        self.reviewer_id: Optional[str] = None
        self.review_time: Optional[datetime] = None

        # 히스토리 정보
        self.modification_history: List[Dict[str, Any]] = []

        # 초기 검증
        self._validate_initial_state()

    def _validate_initial_state(self) -> None:
        """초기 상태 검증"""
        if not self.document_id:
            raise ValidationError(
                message="Document ID cannot be empty", validation_type="document_id"
            )

        if self.page_number < 1:
            raise ValidationError(
                message=f"Invalid page number: {self.page_number}",
                validation_type="page_number",
            )

    # BaseModel 추상 메서드 구현
    def to_dict(self) -> Dict[str, Any]:
        """
        어노테이션 모델을 딕셔너리로 변환

        Returns:
            Dict[str, Any]: 어노테이션 모델 딕셔너리
        """
        return {
            "annotation_id": self.annotation_id,
            "document_id": self.document_id,
            "page_number": self.page_number,
            "annotation_type": self.annotation_type.value,
            "annotation_status": self.annotation_status.value,
            "field_annotations": {
                field_id: field_annotation.to_dict()
                for field_id, field_annotation in self.field_annotations.items()
            },
            "field_count": self.field_count,
            "overall_confidence": self.overall_confidence,
            "overall_quality": self.overall_quality,
            "completion_percentage": self.completion_percentage,
            "session_id": self.session_id,
            "annotator_id": self.annotator_id,
            "annotation_time": (
                self.annotation_time.isoformat() if self.annotation_time else None
            ),
            "validation_results": self.validation_results,
            "reviewer_id": self.reviewer_id,
            "review_time": self.review_time.isoformat() if self.review_time else None,
            "modification_history": self.modification_history,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "is_valid": self.is_valid,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationModel":
        """
        딕셔너리에서 어노테이션 모델 인스턴스 생성

        Args:
            data: 어노테이션 모델 데이터 딕셔너리

        Returns:
            AnnotationModel: 어노테이션 모델 인스턴스
        """
        # 기본 정보로 인스턴스 생성
        instance = cls(
            document_id=data["document_id"],
            page_number=data["page_number"],
            annotation_type=AnnotationType(data["annotation_type"]),
        )

        # 추가 속성 복원
        instance.annotation_id = data["annotation_id"]
        instance.annotation_status = AnnotationStatus(data["annotation_status"])

        # 필드 어노테이션 복원
        for field_id, field_data in data.get("field_annotations", {}).items():
            instance.field_annotations[field_id] = FieldAnnotation.from_dict(field_data)

        instance.field_count = data.get("field_count", 0)
        instance.overall_confidence = data.get("overall_confidence", 0.0)
        instance.overall_quality = data.get("overall_quality", 0.0)
        instance.completion_percentage = data.get("completion_percentage", 0.0)

        # 세션 정보 복원
        instance.session_id = data.get("session_id")
        instance.annotator_id = data.get("annotator_id")
        if data.get("annotation_time"):
            instance.annotation_time = datetime.fromisoformat(data["annotation_time"])

        # 검증 정보 복원
        instance.validation_results = data.get("validation_results", {})
        instance.reviewer_id = data.get("reviewer_id")
        if data.get("review_time"):
            instance.review_time = datetime.fromisoformat(data["review_time"])

        # 히스토리 복원
        instance.modification_history = data.get("modification_history", [])

        return instance

    def validate(self) -> bool:
        """
        어노테이션 모델 데이터 유효성 검증

        Returns:
            bool: 유효성 검증 결과
        """
        self.clear_validation_errors()

        try:
            # 기본 정보 검증
            if not self.document_id:
                self.add_validation_error("Document ID is required")

            if self.page_number < 1:
                self.add_validation_error("Page number must be positive")

            # 필드 어노테이션 검증
            valid_fields = 0
            for field_id, field_annotation in self.field_annotations.items():
                if field_annotation.validate_field_value():
                    valid_fields += 1
                else:
                    self.add_validation_error(f"Field validation failed: {field_id}")

            # 완성도 검증
            if self.field_count > 0:
                self.completion_percentage = (valid_fields / self.field_count) * 100

            # 품질 점수 검증
            if self.overall_quality < ANNOTATION_QUALITY_THRESHOLD:
                self.add_validation_error(
                    f"Overall quality too low: {self.overall_quality}"
                )

            # 검증 완료
            self.is_valid = len(self.validation_errors) == 0
            return self.is_valid

        except Exception as e:
            self.add_validation_error(f"Validation failed: {str(e)}")
            return False

    # 추가 메서드들
    def add_field_annotation(
        self,
        field_name: str,
        bounding_box: BoundingBox,
        text_value: str,
        field_type: AnnotationType = AnnotationType.TEXT,
        confidence_score: float = 1.0,
    ) -> str:
        """
        필드 어노테이션 추가

        Args:
            field_name: 필드 이름
            bounding_box: 바운딩 박스
            text_value: 텍스트 값
            field_type: 필드 타입
            confidence_score: 신뢰도 점수

        Returns:
            str: 생성된 필드 ID
        """
        field_annotation = FieldAnnotation(
            field_id=str(uuid.uuid4()),
            field_name=field_name,
            field_type=field_type,
            bounding_box=bounding_box,
            text_value=text_value,
            original_text=text_value,
            confidence_score=confidence_score,
        )

        self.field_annotations[field_annotation.field_id] = field_annotation
        self.field_count = len(self.field_annotations)

        # 전체 품질 점수 재계산
        self._recalculate_overall_scores()

        # 히스토리 추가
        self._add_modification_history(
            action="add_field",
            field_id=field_annotation.field_id,
            field_name=field_name,
        )

        self._updated_at = datetime.now()
        self._version += 1

        return field_annotation.field_id

    def remove_field_annotation(self, field_id: str) -> bool:
        """
        필드 어노테이션 제거

        Args:
            field_id: 필드 ID

        Returns:
            bool: 제거 성공 여부
        """
        if field_id in self.field_annotations:
            field_annotation = self.field_annotations[field_id]
            del self.field_annotations[field_id]
            self.field_count = len(self.field_annotations)

            # 전체 품질 점수 재계산
            self._recalculate_overall_scores()

            # 히스토리 추가
            self._add_modification_history(
                action="remove_field",
                field_id=field_id,
                field_name=field_annotation.field_name,
            )

            self._updated_at = datetime.now()
            self._version += 1

            return True

        return False

    def update_field_annotation(
        self,
        field_id: str,
        text_value: Optional[str] = None,
        bounding_box: Optional[BoundingBox] = None,
        confidence_score: Optional[float] = None,
    ) -> bool:
        """
        필드 어노테이션 업데이트

        Args:
            field_id: 필드 ID
            text_value: 새로운 텍스트 값
            bounding_box: 새로운 바운딩 박스
            confidence_score: 새로운 신뢰도 점수

        Returns:
            bool: 업데이트 성공 여부
        """
        if field_id not in self.field_annotations:
            return False

        field_annotation = self.field_annotations[field_id]

        # 업데이트 정보 저장
        updates = {}

        if text_value is not None:
            field_annotation.update_value(text_value)
            updates["text_value"] = text_value

        if bounding_box is not None:
            field_annotation.set_bounding_box(bounding_box)
            updates["bounding_box"] = bounding_box.to_dict()

        if confidence_score is not None:
            field_annotation.confidence_score = confidence_score
            updates["confidence_score"] = confidence_score

        # 전체 품질 점수 재계산
        self._recalculate_overall_scores()

        # 히스토리 추가
        self._add_modification_history(
            action="update_field", field_id=field_id, updates=updates
        )

        self._updated_at = datetime.now()
        self._version += 1

        return True

    def get_field_annotation(self, field_id: str) -> Optional[FieldAnnotation]:
        """
        필드 어노테이션 조회

        Args:
            field_id: 필드 ID

        Returns:
            Optional[FieldAnnotation]: 필드 어노테이션 (없으면 None)
        """
        return self.field_annotations.get(field_id)

    def get_field_annotations_by_type(
        self, field_type: AnnotationType
    ) -> List[FieldAnnotation]:
        """
        타입별 필드 어노테이션 조회

        Args:
            field_type: 필드 타입

        Returns:
            List[FieldAnnotation]: 해당 타입의 필드 어노테이션 목록
        """
        return [
            field_annotation
            for field_annotation in self.field_annotations.values()
            if field_annotation.field_type == field_type
        ]

    def get_field_annotations_by_name(self, field_name: str) -> List[FieldAnnotation]:
        """
        이름별 필드 어노테이션 조회

        Args:
            field_name: 필드 이름

        Returns:
            List[FieldAnnotation]: 해당 이름의 필드 어노테이션 목록
        """
        return [
            field_annotation
            for field_annotation in self.field_annotations.values()
            if field_annotation.field_name == field_name
        ]

    def _recalculate_overall_scores(self) -> None:
        """전체 품질 점수 재계산"""
        if not self.field_annotations:
            self.overall_confidence = 0.0
            self.overall_quality = 0.0
            self.completion_percentage = 0.0
            return

        # 신뢰도 점수 계산
        confidence_scores = [
            field.confidence_score for field in self.field_annotations.values()
        ]
        self.overall_confidence = sum(confidence_scores) / len(confidence_scores)

        # 품질 점수 계산
        quality_scores = [
            field.quality_score for field in self.field_annotations.values()
        ]
        self.overall_quality = sum(quality_scores) / len(quality_scores)

        # 완성도 계산
        validated_fields = sum(
            1 for field in self.field_annotations.values() if field.is_validated
        )
        self.completion_percentage = (
            validated_fields / len(self.field_annotations)
        ) * 100

    def _add_modification_history(self, action: str, **kwargs) -> None:
        """수정 히스토리 추가"""
        history_entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "version": self._version,
            **kwargs,
        }
        self.modification_history.append(history_entry)

    def start_annotation_session(
        self, annotator_id: str, session_id: Optional[str] = None
    ) -> str:
        """
        어노테이션 세션 시작

        Args:
            annotator_id: 어노테이터 ID
            session_id: 세션 ID (없으면 자동 생성)

        Returns:
            str: 세션 ID
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        self.session_id = session_id
        self.annotator_id = annotator_id
        self.annotation_time = datetime.now()
        self.annotation_status = AnnotationStatus.IN_PROGRESS

        self._add_modification_history(
            action="start_session", session_id=session_id, annotator_id=annotator_id
        )

        return session_id

    def complete_annotation(self) -> bool:
        """
        어노테이션 완료

        Returns:
            bool: 완료 성공 여부
        """
        if not self.field_annotations:
            return False

        # 모든 필드 검증
        all_valid = all(
            field.validate_field_value() for field in self.field_annotations.values()
        )

        if all_valid:
            self.annotation_status = AnnotationStatus.COMPLETED
            self._add_modification_history(action="complete_annotation")
            return True

        return False

    def submit_for_review(self, reviewer_id: str) -> bool:
        """
        리뷰를 위해 제출

        Args:
            reviewer_id: 리뷰어 ID

        Returns:
            bool: 제출 성공 여부
        """
        if self.annotation_status != AnnotationStatus.COMPLETED:
            return False

        self.reviewer_id = reviewer_id
        self.annotation_status = AnnotationStatus.VALIDATED

        self._add_modification_history(
            action="submit_for_review", reviewer_id=reviewer_id
        )

        return True

    def calculate_annotation_coverage(self) -> float:
        """
        어노테이션 커버리지 계산

        Returns:
            float: 커버리지 비율 (0.0 ~ 1.0)
        """
        if not self.field_annotations:
            return 0.0

        # 바운딩 박스 면적 합계 계산
        total_area = sum(
            field.bounding_box.area for field in self.field_annotations.values()
        )

        # 전체 페이지 면적 대비 비율 (임시로 A4 크기 기준)
        page_area = 595 * 842  # A4 크기 (포인트 단위)
        coverage = min(total_area / page_area, 1.0)

        return coverage

    def merge_annotations(self, other: "AnnotationModel") -> "AnnotationModel":
        """
        다른 어노테이션과 병합

        Args:
            other: 병합할 다른 어노테이션 모델

        Returns:
            AnnotationModel: 병합된 어노테이션 모델
        """
        if (
            self.document_id != other.document_id
            or self.page_number != other.page_number
        ):
            raise ValidationError(
                message="Cannot merge annotations from different documents or pages",
                validation_type="merge_compatibility",
            )

        # 새로운 인스턴스 생성
        merged = AnnotationModel(
            document_id=self.document_id,
            page_number=self.page_number,
            annotation_type=self.annotation_type,
        )

        # 필드 어노테이션 병합
        for field_id, field_annotation in self.field_annotations.items():
            merged.field_annotations[field_id] = field_annotation

        for field_id, field_annotation in other.field_annotations.items():
            if field_id not in merged.field_annotations:
                merged.field_annotations[field_id] = field_annotation

        merged.field_count = len(merged.field_annotations)
        merged._recalculate_overall_scores()

        # 히스토리 병합
        merged.modification_history = (
            self.modification_history + other.modification_history
        )
        merged.modification_history.sort(key=lambda x: x["timestamp"])

        return merged

    def export_to_coco_format(self) -> Dict[str, Any]:
        """
        COCO 포맷으로 내보내기

        Returns:
            Dict[str, Any]: COCO 포맷 데이터
        """
        annotations = []

        for field_id, field_annotation in self.field_annotations.items():
            bbox = field_annotation.bounding_box

            annotation = {
                "id": field_id,
                "category_id": field_annotation.field_type.value,
                "bbox": [bbox.x, bbox.y, bbox.width, bbox.height],
                "area": bbox.area,
                "segmentation": [],
                "iscrowd": 0,
                "attributes": {
                    "text": field_annotation.text_value,
                    "confidence": field_annotation.confidence_score,
                    "field_name": field_annotation.field_name,
                },
            }

            annotations.append(annotation)

        return {
            "annotations": annotations,
            "image_id": f"{self.document_id}_{self.page_number}",
            "categories": [
                {"id": ann_type.value, "name": ann_type.name}
                for ann_type in AnnotationType
            ],
        }

    def clone(self) -> "AnnotationModel":
        """
        어노테이션 모델 복제

        Returns:
            AnnotationModel: 복제된 어노테이션 모델
        """
        cloned_dict = self.to_dict()
        cloned_dict["annotation_id"] = str(uuid.uuid4())  # 새로운 ID 생성
        return self.from_dict(cloned_dict)

    def __str__(self) -> str:
        """문자열 표현"""
        return f"AnnotationModel(id={self.annotation_id}, doc={self.document_id}, page={self.page_number}, fields={self.field_count})"

    def __repr__(self) -> str:
        """개발자용 표현"""
        return (
            f"AnnotationModel("
            f"annotation_id='{self.annotation_id}', "
            f"document_id='{self.document_id}', "
            f"page_number={self.page_number}, "
            f"field_count={self.field_count}, "
            f"status={self.annotation_status.name}, "
            f"completion={self.completion_percentage:.1f}%"
            f")"
        )


# ====================================================================================
# 4. 문서 어노테이션 클래스
# ====================================================================================


class DocumentAnnotation(BaseModel):
    """
    문서 어노테이션 클래스

    전체 문서의 어노테이션을 관리합니다.
    """

    def __init__(self, document_id: str):
        """
        DocumentAnnotation 초기화

        Args:
            document_id: 문서 ID
        """
        super().__init__()

        self.document_id = document_id
        self.page_annotations: Dict[int, AnnotationModel] = {}
        self.document_metadata: Dict[str, Any] = {}
        self.annotation_schema: Dict[str, Any] = {}
        self.completion_status = "pending"
        self.total_fields = 0
        self.completed_fields = 0

    def add_page_annotation(
        self, page_number: int, annotation: AnnotationModel
    ) -> None:
        """페이지 어노테이션 추가"""
        self.page_annotations[page_number] = annotation
        self._update_completion_status()

    def get_page_annotation(self, page_number: int) -> Optional[AnnotationModel]:
        """페이지 어노테이션 조회"""
        return self.page_annotations.get(page_number)

    def _update_completion_status(self) -> None:
        """완성도 상태 업데이트"""
        self.total_fields = sum(
            len(annotation.field_annotations)
            for annotation in self.page_annotations.values()
        )

        self.completed_fields = sum(
            sum(
                1
                for field in annotation.field_annotations.values()
                if field.is_validated
            )
            for annotation in self.page_annotations.values()
        )

        if self.total_fields > 0:
            completion_rate = self.completed_fields / self.total_fields
            if completion_rate >= 1.0:
                self.completion_status = "completed"
            elif completion_rate >= 0.5:
                self.completion_status = "in_progress"
            else:
                self.completion_status = "pending"

    def to_dict(self) -> Dict[str, Any]:
        """문서 어노테이션을 딕셔너리로 변환"""
        return {
            "document_id": self.document_id,
            "page_annotations": {
                str(page_num): annotation.to_dict()
                for page_num, annotation in self.page_annotations.items()
            },
            "document_metadata": self.document_metadata,
            "annotation_schema": self.annotation_schema,
            "completion_status": self.completion_status,
            "total_fields": self.total_fields,
            "completed_fields": self.completed_fields,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentAnnotation":
        """딕셔너리에서 문서 어노테이션 인스턴스 생성"""
        instance = cls(document_id=data["document_id"])

        # 페이지 어노테이션 복원
        for page_num_str, annotation_data in data.get("page_annotations", {}).items():
            page_num = int(page_num_str)
            annotation = AnnotationModel.from_dict(annotation_data)
            instance.page_annotations[page_num] = annotation

        instance.document_metadata = data.get("document_metadata", {})
        instance.annotation_schema = data.get("annotation_schema", {})
        instance.completion_status = data.get("completion_status", "pending")
        instance.total_fields = data.get("total_fields", 0)
        instance.completed_fields = data.get("completed_fields", 0)

        return instance

    def validate(self) -> bool:
        """문서 어노테이션 검증"""
        self.clear_validation_errors()

        if not self.document_id:
            self.add_validation_error("Document ID is required")
            return False

        # 각 페이지 어노테이션 검증
        for page_num, annotation in self.page_annotations.items():
            if not annotation.validate():
                self.add_validation_error(
                    f"Page {page_num} annotation validation failed"
                )

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid


# ====================================================================================
# 5. 어노테이션 컬렉션 클래스
# ====================================================================================


class AnnotationCollection(BaseModel):
    """
    어노테이션 컬렉션 클래스

    여러 문서의 어노테이션을 관리합니다.
    """

    def __init__(self, collection_name: str):
        """
        AnnotationCollection 초기화

        Args:
            collection_name: 컬렉션 이름
        """
        super().__init__()

        self.collection_id = str(uuid.uuid4())
        self.collection_name = collection_name
        self.document_annotations: Dict[str, DocumentAnnotation] = {}
        self.collection_metadata: Dict[str, Any] = {}
        self.statistics: Dict[str, Any] = {}

    def add_document_annotation(self, document_annotation: DocumentAnnotation) -> None:
        """문서 어노테이션 추가"""
        self.document_annotations[document_annotation.document_id] = document_annotation
        self._update_statistics()

    def get_document_annotation(self, document_id: str) -> Optional[DocumentAnnotation]:
        """문서 어노테이션 조회"""
        return self.document_annotations.get(document_id)

    def _update_statistics(self) -> None:
        """통계 정보 업데이트"""
        self.statistics = {
            "total_documents": len(self.document_annotations),
            "total_pages": sum(
                len(doc.page_annotations) for doc in self.document_annotations.values()
            ),
            "total_annotations": sum(
                sum(
                    len(annotation.field_annotations)
                    for annotation in doc.page_annotations.values()
                )
                for doc in self.document_annotations.values()
            ),
            "completion_rate": self._calculate_completion_rate(),
        }

    def _calculate_completion_rate(self) -> float:
        """완성도 계산"""
        if not self.document_annotations:
            return 0.0

        completed_docs = sum(
            1
            for doc in self.document_annotations.values()
            if doc.completion_status == "completed"
        )

        return completed_docs / len(self.document_annotations)

    def to_dict(self) -> Dict[str, Any]:
        """컬렉션을 딕셔너리로 변환"""
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "document_annotations": {
                doc_id: doc_annotation.to_dict()
                for doc_id, doc_annotation in self.document_annotations.items()
            },
            "collection_metadata": self.collection_metadata,
            "statistics": self.statistics,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationCollection":
        """딕셔너리에서 컬렉션 인스턴스 생성"""
        instance = cls(collection_name=data["collection_name"])
        instance.collection_id = data["collection_id"]

        # 문서 어노테이션 복원
        for doc_id, doc_data in data.get("document_annotations", {}).items():
            instance.document_annotations[doc_id] = DocumentAnnotation.from_dict(
                doc_data
            )

        instance.collection_metadata = data.get("collection_metadata", {})
        instance.statistics = data.get("statistics", {})

        return instance

    def validate(self) -> bool:
        """컬렉션 검증"""
        self.clear_validation_errors()

        if not self.collection_name:
            self.add_validation_error("Collection name is required")
            return False

        # 각 문서 어노테이션 검증
        for doc_id, doc_annotation in self.document_annotations.items():
            if not doc_annotation.validate():
                self.add_validation_error(
                    f"Document {doc_id} annotation validation failed"
                )

        self.is_valid = len(self.validation_errors) == 0
        return self.is_valid


# ====================================================================================
# 6. 유틸리티 함수들
# ====================================================================================


def create_bounding_box(x: int, y: int, width: int, height: int) -> BoundingBox:
    """
    바운딩 박스 생성

    Args:
        x: X 좌표
        y: Y 좌표
        width: 너비
        height: 높이

    Returns:
        BoundingBox: 생성된 바운딩 박스
    """
    return BoundingBox(x=x, y=y, width=width, height=height)


def create_field_annotation(
    field_name: str,
    bbox: BoundingBox,
    text_value: str,
    field_type: AnnotationType = AnnotationType.TEXT,
) -> FieldAnnotation:
    """
    필드 어노테이션 생성

    Args:
        field_name: 필드 이름
        bbox: 바운딩 박스
        text_value: 텍스트 값
        field_type: 필드 타입

    Returns:
        FieldAnnotation: 생성된 필드 어노테이션
    """
    return FieldAnnotation(
        field_id=str(uuid.uuid4()),
        field_name=field_name,
        field_type=field_type,
        bounding_box=bbox,
        text_value=text_value,
        original_text=text_value,
    )


def validate_annotation_batch(annotations: List[AnnotationModel]) -> Dict[str, Any]:
    """
    어노테이션 배치 검증

    Args:
        annotations: 검증할 어노테이션 목록

    Returns:
        Dict[str, Any]: 배치 검증 결과
    """
    results = {
        "total_annotations": len(annotations),
        "valid_annotations": 0,
        "invalid_annotations": 0,
        "validation_errors": [],
        "statistics": {
            "total_fields": 0,
            "validated_fields": 0,
            "average_confidence": 0.0,
            "average_quality": 0.0,
        },
    }

    total_confidence = 0.0
    total_quality = 0.0

    for annotation in annotations:
        if annotation.validate():
            results["valid_annotations"] += 1
        else:
            results["invalid_annotations"] += 1
            results["validation_errors"].extend(annotation.validation_errors)

        # 통계 수집
        results["statistics"]["total_fields"] += annotation.field_count
        results["statistics"]["validated_fields"] += sum(
            1 for field in annotation.field_annotations.values() if field.is_validated
        )

        total_confidence += annotation.overall_confidence
        total_quality += annotation.overall_quality

    if annotations:
        results["statistics"]["average_confidence"] = total_confidence / len(
            annotations
        )
        results["statistics"]["average_quality"] = total_quality / len(annotations)

    return results


def export_annotations_to_json(
    annotations: List[AnnotationModel], output_path: str
) -> bool:
    """
    어노테이션을 JSON 파일로 내보내기

    Args:
        annotations: 내보낼 어노테이션 목록
        output_path: 출력 파일 경로

    Returns:
        bool: 내보내기 성공 여부
    """
    try:
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_annotations": len(annotations),
                "format_version": "1.0",
            },
            "annotations": [annotation.to_dict() for annotation in annotations],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        return True

    except Exception as e:
        raise ProcessingError(
            message=f"Failed to export annotations to JSON: {str(e)}",
            original_exception=e,
        )


def import_annotations_from_json(json_path: str) -> List[AnnotationModel]:
    """
    JSON 파일에서 어노테이션 가져오기

    Args:
        json_path: JSON 파일 경로

    Returns:
        List[AnnotationModel]: 가져온 어노테이션 목록
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = []
        for annotation_data in data.get("annotations", []):
            annotation = AnnotationModel.from_dict(annotation_data)
            annotations.append(annotation)

        return annotations

    except Exception as e:
        raise ProcessingError(
            message=f"Failed to import annotations from JSON: {str(e)}",
            original_exception=e,
        )


if __name__ == "__main__":
    # 어노테이션 모델 테스트
    print("YOKOGAWA OCR 어노테이션 모델 테스트")
    print("=" * 50)

    try:
        # 바운딩 박스 생성 테스트
        bbox = create_bounding_box(100, 100, 200, 50)
        print(f"✅ 바운딩 박스 생성: {bbox}")

        # 필드 어노테이션 생성 테스트
        field_annotation = create_field_annotation(
            field_name="company_name",
            bbox=bbox,
            text_value="YOKOGAWA Electric Corporation",
            field_type=AnnotationType.TEXT,
        )
        print(f"✅ 필드 어노테이션 생성: {field_annotation.field_id}")

        # 어노테이션 모델 생성 테스트
        annotation_model = AnnotationModel(
            document_id="test_doc_001",
            page_number=1,
            annotation_type=AnnotationType.TEXT,
        )

        # 필드 추가 테스트
        field_id = annotation_model.add_field_annotation(
            field_name="company_name",
            bounding_box=bbox,
            text_value="YOKOGAWA Electric Corporation",
            field_type=AnnotationType.TEXT,
        )
        print(f"✅ 필드 추가 완료: {field_id}")

        # 검증 테스트
        is_valid = annotation_model.validate()
        print(f"✅ 어노테이션 검증 결과: {'통과' if is_valid else '실패'}")

        # 딕셔너리 변환 테스트
        annotation_dict = annotation_model.to_dict()
        print(
            f"✅ 딕셔너리 변환 완료 (필드 수: {len(annotation_dict['field_annotations'])})"
        )

        # 복원 테스트
        restored_annotation = AnnotationModel.from_dict(annotation_dict)
        print(f"✅ 딕셔너리에서 복원 완료: {restored_annotation.annotation_id}")

        # 어노테이션 완료 테스트
        session_id = annotation_model.start_annotation_session("test_annotator")
        completion_success = annotation_model.complete_annotation()
        print(f"✅ 어노테이션 완료: {'성공' if completion_success else '실패'}")

        # 커버리지 계산 테스트
        coverage = annotation_model.calculate_annotation_coverage()
        print(f"✅ 어노테이션 커버리지: {coverage:.2%}")

    except Exception as e:
        print(f"❌ 어노테이션 모델 테스트 실패: {e}")

    print("\n🎯 어노테이션 모델 구현이 완료되었습니다!")
