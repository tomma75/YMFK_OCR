#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 모델 패키지 초기화 모듈

이 모듈은 전체 시스템의 데이터 모델을 관리합니다.
문서 모델, 어노테이션 모델 등 모든 데이터 구조를 중앙에서 관리합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
버전: 1.0.0
"""

# ====================================================================================
# 1. 패키지 메타데이터
# ====================================================================================

__version__ = "1.0.0"
__author__ = "YOKOGAWA OCR 개발팀"
__email__ = "ocr-dev@yokogawa.com"
__description__ = "YOKOGAWA OCR 데이터 준비 프로젝트 - 데이터 모델 패키지"
__license__ = "YOKOGAWA Proprietary"

# ====================================================================================
# 2. 문서 모델 관련 클래스 및 열거형 (document_model.py)
# ====================================================================================

from .document_model import (
    # 문서 관련 열거형
    DocumentType,
    DocumentStatus,
    PageType,
    # 문서 메타데이터 클래스
    DocumentMetadata,
    # 페이지 정보 클래스
    PageInfo,
    # 메인 문서 모델 클래스
    DocumentModel,
    # 문서 통계 클래스
    DocumentStatistics,
)

# ====================================================================================
# 3. 어노테이션 모델 관련 클래스 및 열거형 (annotation_model.py)
# ====================================================================================

from .annotation_model import (
    # 어노테이션 관련 열거형
    AnnotationType,
    AnnotationStatus,
    CoordinateSystem,
    # 바운딩 박스 클래스
    BoundingBox,
    # 필드 어노테이션 클래스
    FieldAnnotation,
    # 메인 어노테이션 모델 클래스
    AnnotationModel,
    # 문서 어노테이션 클래스
    DocumentAnnotation,
    # 어노테이션 컬렉션 클래스
    AnnotationCollection,
)

# ====================================================================================
# 4. 실제 파일에 정의된 함수들을 사용한 유틸리티 함수들
# ====================================================================================

from typing import Any, Dict, List, Optional, Union, Type, TypeVar

# 타입 변수 정의
ModelType = TypeVar("ModelType", bound="BaseModel")
DocumentModelType = TypeVar("DocumentModelType", bound=DocumentModel)
AnnotationModelType = TypeVar("AnnotationModelType", bound=AnnotationModel)


def create_bounding_box_from_coordinates(
    x: int, y: int, width: int, height: int
) -> BoundingBox:
    """
    좌표로부터 바운딩 박스 생성

    Args:
        x: X 좌표
        y: Y 좌표
        width: 너비
        height: 높이

    Returns:
        BoundingBox: 생성된 바운딩 박스 인스턴스
    """
    return BoundingBox(x=x, y=y, width=width, height=height)


def create_document_model_from_file(file_path: str) -> DocumentModel:
    """
    파일 경로로부터 문서 모델 생성

    Args:
        file_path: 파일 경로

    Returns:
        DocumentModel: 생성된 문서 모델 인스턴스
    """
    return DocumentModel.from_file_path(file_path)


def validate_document_model(document: DocumentModel) -> bool:
    """
    문서 모델 유효성 검증

    Args:
        document: 검증할 문서 모델

    Returns:
        bool: 검증 결과
    """
    return document.validate_document_format()


def extract_document_text(document: DocumentModel) -> str:
    """
    문서에서 텍스트 추출

    Args:
        document: 문서 모델

    Returns:
        str: 추출된 텍스트
    """
    return document.extract_text_content()


def get_document_page_count(document: DocumentModel) -> int:
    """
    문서 페이지 수 조회

    Args:
        document: 문서 모델

    Returns:
        int: 페이지 수
    """
    return document.get_page_count()


def validate_annotation_model(annotation: AnnotationModel) -> bool:
    """
    어노테이션 모델 유효성 검증

    Args:
        annotation: 검증할 어노테이션 모델

    Returns:
        bool: 검증 결과
    """
    return annotation.validate_annotation_structure()


def calculate_annotation_model_coverage(annotation: AnnotationModel) -> float:
    """
    어노테이션 모델 커버리지 계산

    Args:
        annotation: 어노테이션 모델

    Returns:
        float: 커버리지 비율 (0.0 ~ 1.0)
    """
    return annotation.calculate_annotation_coverage()


def merge_annotation_models(
    primary_annotation: AnnotationModel, secondary_annotation: AnnotationModel
) -> AnnotationModel:
    """
    어노테이션 모델 병합

    Args:
        primary_annotation: 기본 어노테이션
        secondary_annotation: 병합할 어노테이션

    Returns:
        AnnotationModel: 병합된 어노테이션
    """
    return primary_annotation.merge_annotations(secondary_annotation)


def create_field_annotation_with_bbox(
    field_name: str,
    bbox: BoundingBox,
    text_value: str,
    field_type: AnnotationType = AnnotationType.TEXT,
) -> FieldAnnotation:
    """
    바운딩 박스를 사용한 필드 어노테이션 생성

    Args:
        field_name: 필드 이름
        bbox: 바운딩 박스
        text_value: 텍스트 값
        field_type: 필드 타입

    Returns:
        FieldAnnotation: 생성된 필드 어노테이션
    """
    import uuid

    field_annotation = FieldAnnotation(
        field_id=str(uuid.uuid4()),
        field_name=field_name,
        field_type=field_type,
        bounding_box=bbox,
        text_value=text_value,
    )
    return field_annotation


def validate_field_annotation(field: FieldAnnotation) -> bool:
    """
    필드 어노테이션 유효성 검증

    Args:
        field: 필드 어노테이션

    Returns:
        bool: 검증 결과
    """
    return field.validate_field_value()


def update_field_annotation_value(field: FieldAnnotation, new_value: str) -> None:
    """
    필드 어노테이션 값 업데이트

    Args:
        field: 필드 어노테이션
        new_value: 새로운 값
    """
    field.update_value(new_value)


def convert_model_to_dict(
    model: Union[DocumentModel, AnnotationModel],
) -> Dict[str, Any]:
    """
    모델을 딕셔너리로 변환

    Args:
        model: 변환할 모델

    Returns:
        Dict[str, Any]: 딕셔너리 데이터
    """
    return model.to_dict()


def convert_models_to_dict_list(
    models: List[Union[DocumentModel, AnnotationModel]],
) -> List[Dict[str, Any]]:
    """
    모델 목록을 딕셔너리 목록으로 변환

    Args:
        models: 모델 목록

    Returns:
        List[Dict[str, Any]]: 딕셔너리 목록
    """
    return [model.to_dict() for model in models]


def create_document_model_from_dict(data: Dict[str, Any]) -> DocumentModel:
    """
    딕셔너리에서 문서 모델 생성

    Args:
        data: 문서 데이터

    Returns:
        DocumentModel: 생성된 문서 모델
    """
    return DocumentModel.from_dict(data)


def create_annotation_model_from_dict(data: Dict[str, Any]) -> AnnotationModel:
    """
    딕셔너리에서 어노테이션 모델 생성

    Args:
        data: 어노테이션 데이터

    Returns:
        AnnotationModel: 생성된 어노테이션 모델
    """
    return AnnotationModel.from_dict(data)


def get_supported_document_types() -> List[str]:
    """
    지원되는 문서 타입 목록 반환

    Returns:
        List[str]: 지원되는 문서 타입 목록
    """
    return [doc_type.value for doc_type in DocumentType]


def get_supported_annotation_types() -> List[str]:
    """
    지원되는 어노테이션 타입 목록 반환

    Returns:
        List[str]: 지원되는 어노테이션 타입 목록
    """
    return [ann_type.value for ann_type in AnnotationType]


def get_supported_annotation_statuses() -> List[str]:
    """
    지원되는 어노테이션 상태 목록 반환

    Returns:
        List[str]: 지원되는 어노테이션 상태 목록
    """
    return [status.value for status in AnnotationStatus]


def get_supported_document_statuses() -> List[str]:
    """
    지원되는 문서 상태 목록 반환

    Returns:
        List[str]: 지원되는 문서 상태 목록
    """
    return [status.value for status in DocumentStatus]


def filter_models_by_document_status(
    models: List[DocumentModel], status: DocumentStatus
) -> List[DocumentModel]:
    """
    문서 상태별 모델 필터링

    Args:
        models: 문서 모델 목록
        status: 필터링할 상태

    Returns:
        List[DocumentModel]: 필터링된 문서 모델 목록
    """
    return [
        model
        for model in models
        if hasattr(model, "document_status") and model.document_status == status
    ]


def filter_models_by_annotation_status(
    models: List[AnnotationModel], status: AnnotationStatus
) -> List[AnnotationModel]:
    """
    어노테이션 상태별 모델 필터링

    Args:
        models: 어노테이션 모델 목록
        status: 필터링할 상태

    Returns:
        List[AnnotationModel]: 필터링된 어노테이션 모델 목록
    """
    return [
        model
        for model in models
        if hasattr(model, "annotation_status") and model.annotation_status == status
    ]


def get_bounding_box_corners(bbox: BoundingBox) -> List[tuple]:
    """
    바운딩 박스 모서리 좌표 반환

    Args:
        bbox: 바운딩 박스

    Returns:
        List[tuple]: 모서리 좌표 목록
    """
    return bbox.get_corners()


def check_bounding_box_contains_point(bbox: BoundingBox, x: int, y: int) -> bool:
    """
    바운딩 박스 내부에 점이 있는지 확인

    Args:
        bbox: 바운딩 박스
        x: X 좌표
        y: Y 좌표

    Returns:
        bool: 포함 여부
    """
    return bbox.contains_point(x, y)


def check_bounding_boxes_intersect(bbox1: BoundingBox, bbox2: BoundingBox) -> bool:
    """
    두 바운딩 박스가 교차하는지 확인

    Args:
        bbox1: 첫 번째 바운딩 박스
        bbox2: 두 번째 바운딩 박스

    Returns:
        bool: 교차 여부
    """
    return bbox1.intersects_with(bbox2)


def calculate_bounding_box_intersection_area(
    bbox1: BoundingBox, bbox2: BoundingBox
) -> int:
    """
    두 바운딩 박스의 교차 면적 계산

    Args:
        bbox1: 첫 번째 바운딩 박스
        bbox2: 두 번째 바운딩 박스

    Returns:
        int: 교차 면적
    """
    return bbox1.intersection_area(bbox2)


def calculate_bounding_box_iou(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
    """
    두 바운딩 박스의 IoU 계산

    Args:
        bbox1: 첫 번째 바운딩 박스
        bbox2: 두 번째 바운딩 박스

    Returns:
        float: IoU 값
    """
    return bbox1.iou(bbox2)


def scale_bounding_box(
    bbox: BoundingBox, scale_x: float, scale_y: float
) -> BoundingBox:
    """
    바운딩 박스 크기 조정

    Args:
        bbox: 바운딩 박스
        scale_x: X 축 스케일
        scale_y: Y 축 스케일

    Returns:
        BoundingBox: 조정된 바운딩 박스
    """
    return bbox.scale(scale_x, scale_y)


def normalize_bounding_box(
    bbox: BoundingBox, image_width: int, image_height: int
) -> BoundingBox:
    """
    바운딩 박스 정규화

    Args:
        bbox: 바운딩 박스
        image_width: 이미지 너비
        image_height: 이미지 높이

    Returns:
        BoundingBox: 정규화된 바운딩 박스
    """
    return bbox.normalize(image_width, image_height)


# ====================================================================================
# 5. 모델 통계 및 분석 함수
# ====================================================================================


def get_model_statistics(
    models: List[Union[DocumentModel, AnnotationModel]],
) -> Dict[str, Any]:
    """
    모델 통계 정보 생성

    Args:
        models: 모델 목록

    Returns:
        Dict[str, Any]: 통계 정보
    """
    if not models:
        return {"total_models": 0}

    # 모델 타입별 집계
    type_counts = {}
    for model in models:
        model_type = type(model).__name__
        type_counts[model_type] = type_counts.get(model_type, 0) + 1

    # 유효성 검증 상태 집계
    valid_count = 0
    for model in models:
        if hasattr(model, "is_valid") and model.is_valid:
            valid_count += 1
        elif hasattr(model, "validate"):
            if isinstance(model, DocumentModel):
                if model.validate_document_format():
                    valid_count += 1
            elif isinstance(model, AnnotationModel):
                if model.validate_annotation_structure():
                    valid_count += 1

    invalid_count = len(models) - valid_count

    return {
        "total_models": len(models),
        "model_types": type_counts,
        "valid_models": valid_count,
        "invalid_models": invalid_count,
        "validation_rate": (valid_count / len(models)) * 100 if models else 0,
    }


def check_model_consistency(
    models: List[Union[DocumentModel, AnnotationModel]],
) -> List[str]:
    """
    모델 일관성 검사

    Args:
        models: 모델 목록

    Returns:
        List[str]: 일관성 오류 목록
    """
    consistency_errors = []

    for i, model in enumerate(models):
        try:
            if isinstance(model, DocumentModel):
                if not model.validate_document_format():
                    consistency_errors.append(f"Document model {i} validation failed")
            elif isinstance(model, AnnotationModel):
                if not model.validate_annotation_structure():
                    consistency_errors.append(f"Annotation model {i} validation failed")
        except Exception as e:
            consistency_errors.append(f"Model {i} consistency check failed: {str(e)}")

    return consistency_errors


def get_model_quality_score(model: Union[DocumentModel, AnnotationModel]) -> float:
    """
    모델 품질 점수 계산

    Args:
        model: 모델 인스턴스

    Returns:
        float: 품질 점수 (0.0 ~ 1.0)
    """
    base_score = 0.5

    try:
        # 유효성 검사
        if isinstance(model, DocumentModel):
            if model.validate_document_format():
                base_score += 0.3
        elif isinstance(model, AnnotationModel):
            if model.validate_annotation_structure():
                base_score += 0.3

        # 메타데이터 완성도 확인
        if hasattr(model, "metadata") and model.metadata:
            base_score += 0.1

        # 추가 품질 지표들
        if hasattr(model, "quality_score") and model.quality_score:
            base_score += model.quality_score * 0.1

    except Exception:
        base_score = 0.1

    return min(1.0, base_score)


# ====================================================================================
# 6. 패키지 초기화 및 설정 함수
# ====================================================================================


def initialize_models_package() -> bool:
    """
    모델 패키지 초기화

    Returns:
        bool: 초기화 성공 여부
    """
    try:
        # 문서 모델 클래스 검증
        document_model_classes = [
            DocumentModel,
            DocumentMetadata,
            PageInfo,
            DocumentStatistics,
        ]
        for model_class in document_model_classes:
            if not hasattr(model_class, "to_dict"):
                raise AttributeError(
                    f"{model_class.__name__} must implement to_dict method"
                )

        # 어노테이션 모델 클래스 검증
        annotation_model_classes = [
            AnnotationModel,
            BoundingBox,
            FieldAnnotation,
            DocumentAnnotation,
        ]
        for model_class in annotation_model_classes:
            if not hasattr(model_class, "to_dict"):
                raise AttributeError(
                    f"{model_class.__name__} must implement to_dict method"
                )

        # 열거형 클래스 검증
        enum_classes = [DocumentType, DocumentStatus, AnnotationType, AnnotationStatus]
        for enum_class in enum_classes:
            if not hasattr(enum_class, "__members__"):
                raise AttributeError(f"{enum_class.__name__} is not a valid enum class")

        return True

    except Exception as e:
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"Models package initialization failed: {str(e)}")
        return False


def get_package_info() -> Dict[str, Any]:
    """
    패키지 정보 반환

    Returns:
        Dict[str, Any]: 패키지 정보
    """
    return {
        "name": "models",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "document_model_classes": [
            DocumentModel.__name__,
            DocumentMetadata.__name__,
            PageInfo.__name__,
            DocumentStatistics.__name__,
        ],
        "annotation_model_classes": [
            AnnotationModel.__name__,
            BoundingBox.__name__,
            FieldAnnotation.__name__,
            DocumentAnnotation.__name__,
            AnnotationCollection.__name__,
        ],
        "enum_classes": [
            DocumentType.__name__,
            DocumentStatus.__name__,
            PageType.__name__,
            AnnotationType.__name__,
            AnnotationStatus.__name__,
            CoordinateSystem.__name__,
        ],
        "supported_document_types": get_supported_document_types(),
        "supported_annotation_types": get_supported_annotation_types(),
    }


# ====================================================================================
# 7. 타입 별칭 및 상수 정의
# ====================================================================================

# 타입 별칭
DocumentModelInstance = Union[DocumentModel, Type[DocumentModel]]
AnnotationModelInstance = Union[AnnotationModel, Type[AnnotationModel]]
ModelInstance = Union[
    DocumentModel, AnnotationModel, Type[DocumentModel], Type[AnnotationModel]
]

# 상수 정의
SUPPORTED_DOCUMENT_FORMATS = [doc_type.value for doc_type in DocumentType]
SUPPORTED_ANNOTATION_FORMATS = [ann_type.value for ann_type in AnnotationType]

# 기본 설정값
DEFAULT_DOCUMENT_STATUS = DocumentStatus.PENDING
DEFAULT_ANNOTATION_STATUS = AnnotationStatus.PENDING
DEFAULT_COORDINATE_SYSTEM = CoordinateSystem.ABSOLUTE
DEFAULT_PAGE_TYPE = PageType.CONTENT

# ====================================================================================
# 8. 패키지 레벨 __all__ 정의
# ====================================================================================

__all__ = [
    # 패키지 메타데이터
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__license__",
    # 문서 모델 관련
    "DocumentType",
    "DocumentStatus",
    "PageType",
    "DocumentMetadata",
    "PageInfo",
    "DocumentModel",
    "DocumentStatistics",
    # 어노테이션 모델 관련
    "AnnotationType",
    "AnnotationStatus",
    "CoordinateSystem",
    "BoundingBox",
    "FieldAnnotation",
    "AnnotationModel",
    "DocumentAnnotation",
    "AnnotationCollection",
    # 실제 파일 기반 유틸리티 함수
    "create_bounding_box_from_coordinates",
    "create_document_model_from_file",
    "validate_document_model",
    "extract_document_text",
    "get_document_page_count",
    "validate_annotation_model",
    "calculate_annotation_model_coverage",
    "merge_annotation_models",
    "create_field_annotation_with_bbox",
    "validate_field_annotation",
    "update_field_annotation_value",
    "convert_model_to_dict",
    "convert_models_to_dict_list",
    "create_document_model_from_dict",
    "create_annotation_model_from_dict",
    "get_supported_document_types",
    "get_supported_annotation_types",
    "get_supported_annotation_statuses",
    "get_supported_document_statuses",
    "filter_models_by_document_status",
    "filter_models_by_annotation_status",
    "get_bounding_box_corners",
    "check_bounding_box_contains_point",
    "check_bounding_boxes_intersect",
    "calculate_bounding_box_intersection_area",
    "calculate_bounding_box_iou",
    "scale_bounding_box",
    "normalize_bounding_box",
    # 통계 및 분석 함수
    "get_model_statistics",
    "check_model_consistency",
    "get_model_quality_score",
    # 패키지 관리 함수
    "initialize_models_package",
    "get_package_info",
    # 타입 별칭
    "ModelType",
    "DocumentModelType",
    "AnnotationModelType",
    "DocumentModelInstance",
    "AnnotationModelInstance",
    "ModelInstance",
    # 상수
    "SUPPORTED_DOCUMENT_FORMATS",
    "SUPPORTED_ANNOTATION_FORMATS",
    "DEFAULT_DOCUMENT_STATUS",
    "DEFAULT_ANNOTATION_STATUS",
    "DEFAULT_COORDINATE_SYSTEM",
    "DEFAULT_PAGE_TYPE",
]

# ====================================================================================
# 9. 패키지 로드 시 자동 초기화
# ====================================================================================

# 패키지 로드 시 자동으로 초기화 실행
_initialization_result = initialize_models_package()

if not _initialization_result:
    import warnings

    warnings.warn(
        "Models package initialization failed. Some features may not work correctly.",
        RuntimeWarning,
        stacklevel=2,
    )

# 초기화 성공 메시지 (개발 환경에서만 표시)
import os

if os.getenv("YOKOGAWA_OCR_DEBUG", "false").lower() == "true":
    print(f"✅ YOKOGAWA OCR Models Package v{__version__} initialized successfully")
    print(
        f"   - Document Model Classes: {len([DocumentModel, DocumentMetadata, PageInfo, DocumentStatistics])}"
    )
    print(
        f"   - Annotation Model Classes: {len([AnnotationModel, BoundingBox, FieldAnnotation, DocumentAnnotation, AnnotationCollection])}"
    )
    print(
        f"   - Enum Classes: {len([DocumentType, DocumentStatus, PageType, AnnotationType, AnnotationStatus, CoordinateSystem])}"
    )
    print(f"   - Supported Document Types: {len(SUPPORTED_DOCUMENT_FORMATS)}")
    print(f"   - Supported Annotation Types: {len(SUPPORTED_ANNOTATION_FORMATS)}")
