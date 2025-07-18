#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - Core 패키지 초기화 모듈

이 모듈은 전체 시스템의 기반이 되는 핵심 컴포넌트들을 노출합니다.
모든 추상 클래스, 인터페이스, 예외 클래스를 중앙에서 관리합니다.

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
__description__ = "YOKOGAWA OCR 데이터 준비 프로젝트 - 핵심 기반 클래스 및 인터페이스"
__license__ = "YOKOGAWA Proprietary"

# ====================================================================================
# 2. 기반 추상 클래스 및 인터페이스 (base_classes.py)
# ====================================================================================

# 2.1 상태 관리 열거형
from .base_classes import (
    ServiceStatus,
    ProcessingStatus,
)

# 2.2 데이터 클래스
from .base_classes import (
    ServiceMetrics,
)

# 2.3 핵심 추상 클래스
from .base_classes import (
    BaseService,
    BaseModel,
    BaseProcessor,
    BaseValidator,
)

# 2.4 서비스 인터페이스
from .base_classes import (
    DataCollectionInterface,
    LabelingInterface,
    AugmentationInterface,
    ValidationInterface,
)

# 2.5 서비스 관리 유틸리티
from .base_classes import (
    ServiceRegistry,
)

# 2.6 유틸리티 함수들
from .base_classes import (
    create_service_factory,
    validate_service_interface,
    get_base_class_hierarchy,
    is_abstract_implementation_complete,
)

# ====================================================================================
# 3. 예외 클래스 시스템 (exceptions.py)
# ====================================================================================

# 3.1 오류 관리 열거형
from .exceptions import (
    ErrorSeverity,
    ErrorCode,
)

# 3.2 기본 예외 클래스
from .exceptions import (
    ApplicationError,
)

# 3.3 설정 관련 예외 클래스
from .exceptions import (
    ConfigurationError,
    ConfigurationValidationError,
)

# 3.4 서비스 관련 예외 클래스
from .exceptions import (
    ServiceError,
    ServiceDependencyError,
)

# 3.5 데이터 수집 관련 예외 클래스
from .exceptions import (
    DataCollectionError,
    FileAccessError,
    FileFormatError,
)

# 3.6 라벨링 관련 예외 클래스
from .exceptions import (
    LabelingError,
    AnnotationValidationError,
)

# 3.7 데이터 증강 관련 예외 클래스
from .exceptions import (
    AugmentationError,
    ImageProcessingError,
)

# 3.8 검증 관련 예외 클래스
from .exceptions import (
    ValidationError,
    DataIntegrityError,
)

# 3.9 처리 관련 예외 클래스
from .exceptions import (
    ProcessingError,
    BatchProcessingError,
)

# 3.10 파일 처리 관련 예외 클래스
from .exceptions import (
    FileProcessingError,
    PDFProcessingError,
)

# 3.11 예외 관리 유틸리티 함수들
from .exceptions import (
    create_error_from_exception,
    chain_exceptions,
    get_error_summary,
    filter_errors_by_severity,
    group_errors_by_code,
    handle_exceptions,
    validate_error_hierarchy,
)

# ====================================================================================
# 4. 패키지 레벨 변수 정의
# ====================================================================================

# 4.1 지원되는 서비스 타입
SUPPORTED_SERVICE_TYPES = [
    "DataCollectionService",
    "LabelingService",
    "AugmentationService",
    "ValidationService",
]

# 4.2 지원되는 모델 타입
SUPPORTED_MODEL_TYPES = [
    "DocumentModel",
    "AnnotationModel",
    "ValidationModel",
]

# 4.3 지원되는 프로세서 타입
SUPPORTED_PROCESSOR_TYPES = [
    "FileProcessor",
    "ImageProcessor",
    "PDFProcessor",
]

# 4.4 지원되는 검증자 타입
SUPPORTED_VALIDATOR_TYPES = [
    "DataQualityValidator",
    "AnnotationValidator",
    "ConsistencyChecker",
]

# ====================================================================================
# 5. 패키지 초기화 함수
# ====================================================================================


def initialize_core_package() -> bool:
    """
    Core 패키지 초기화 함수

    패키지 로드 시 기본 검증 및 초기화를 수행합니다.

    Returns:
        bool: 초기화 성공 여부
    """
    try:
        # 예외 계층구조 검증
        if not validate_error_hierarchy():
            raise ApplicationError(
                message="Core package initialization failed: Invalid error hierarchy",
                error_code=ErrorCode.UNKNOWN_ERROR,
                severity=ErrorSeverity.CRITICAL,
            )

        # 추상 클래스 검증
        abstract_classes = [BaseService, BaseModel, BaseProcessor, BaseValidator]
        for cls in abstract_classes:
            if not hasattr(cls, "__abstractmethods__"):
                raise ApplicationError(
                    message=f"Core package initialization failed: {cls.__name__} is not properly abstract",
                    error_code=ErrorCode.UNKNOWN_ERROR,
                    severity=ErrorSeverity.CRITICAL,
                )

        # 인터페이스 검증
        interfaces = [
            DataCollectionInterface,
            LabelingInterface,
            AugmentationInterface,
            ValidationInterface,
        ]
        for interface in interfaces:
            if not hasattr(interface, "__abstractmethods__"):
                raise ApplicationError(
                    message=f"Core package initialization failed: {interface.__name__} is not properly abstract",
                    error_code=ErrorCode.UNKNOWN_ERROR,
                    severity=ErrorSeverity.CRITICAL,
                )

        return True

    except Exception as e:
        # 초기화 실패 시 로깅
        import logging

        logger = logging.getLogger(__name__)
        logger.critical(f"Core package initialization failed: {str(e)}")
        return False


def get_package_info() -> dict:
    """
    패키지 정보 반환

    Returns:
        dict: 패키지 정보 딕셔너리
    """
    return {
        "name": "core",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "supported_services": SUPPORTED_SERVICE_TYPES,
        "supported_models": SUPPORTED_MODEL_TYPES,
        "supported_processors": SUPPORTED_PROCESSOR_TYPES,
        "supported_validators": SUPPORTED_VALIDATOR_TYPES,
        "total_abstract_classes": 4,
        "total_interfaces": 4,
        "total_exception_classes": len(
            [
                cls
                for cls in globals().values()
                if isinstance(cls, type) and issubclass(cls, Exception)
            ]
        ),
    }


# ====================================================================================
# 6. 타입 별칭 정의 (타입 힌팅 지원)
# ====================================================================================

from typing import Type, TypeVar, Union, Dict, Any, List

# 6.1 서비스 타입 별칭
ServiceType = TypeVar("ServiceType", bound=BaseService)
ServiceInstanceType = Union[BaseService, Type[BaseService]]

# 6.2 모델 타입 별칭
ModelType = TypeVar("ModelType", bound=BaseModel)
ModelInstanceType = Union[BaseModel, Type[BaseModel]]

# 6.3 프로세서 타입 별칭
ProcessorType = TypeVar("ProcessorType", bound=BaseProcessor)
ProcessorInstanceType = Union[BaseProcessor, Type[BaseProcessor]]

# 6.4 검증자 타입 별칭
ValidatorType = TypeVar("ValidatorType", bound=BaseValidator)
ValidatorInstanceType = Union[BaseValidator, Type[BaseValidator]]

# 6.5 예외 타입 별칭
ExceptionType = TypeVar("ExceptionType", bound=ApplicationError)
ExceptionInstanceType = Union[ApplicationError, Type[ApplicationError]]

# ====================================================================================
# 7. 패키지 레벨 __all__ 정의
# ====================================================================================

__all__ = [
    # 패키지 메타데이터
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__license__",
    # 상태 관리 열거형
    "ServiceStatus",
    "ProcessingStatus",
    # 데이터 클래스
    "ServiceMetrics",
    # 핵심 추상 클래스
    "BaseService",
    "BaseModel",
    "BaseProcessor",
    "BaseValidator",
    # 서비스 인터페이스
    "DataCollectionInterface",
    "LabelingInterface",
    "AugmentationInterface",
    "ValidationInterface",
    # 서비스 관리
    "ServiceRegistry",
    # 유틸리티 함수
    "create_service_factory",
    "validate_service_interface",
    "get_base_class_hierarchy",
    "is_abstract_implementation_complete",
    # 오류 관리 열거형
    "ErrorSeverity",
    "ErrorCode",
    # 예외 클래스
    "ApplicationError",
    "ConfigurationError",
    "ConfigurationValidationError",
    "ServiceError",
    "ServiceDependencyError",
    "DataCollectionError",
    "FileAccessError",
    "FileFormatError",
    "LabelingError",
    "AnnotationValidationError",
    "AugmentationError",
    "ImageProcessingError",
    "ValidationError",
    "DataIntegrityError",
    "ProcessingError",
    "BatchProcessingError",
    "FileProcessingError",
    "PDFProcessingError",
    # 예외 관리 유틸리티
    "create_error_from_exception",
    "chain_exceptions",
    "get_error_summary",
    "filter_errors_by_severity",
    "group_errors_by_code",
    "handle_exceptions",
    "validate_error_hierarchy",
    # 패키지 레벨 변수
    "SUPPORTED_SERVICE_TYPES",
    "SUPPORTED_MODEL_TYPES",
    "SUPPORTED_PROCESSOR_TYPES",
    "SUPPORTED_VALIDATOR_TYPES",
    # 패키지 함수
    "initialize_core_package",
    "get_package_info",
    # 타입 별칭
    "ServiceType",
    "ServiceInstanceType",
    "ModelType",
    "ModelInstanceType",
    "ProcessorType",
    "ProcessorInstanceType",
    "ValidatorType",
    "ValidatorInstanceType",
    "ExceptionType",
    "ExceptionInstanceType",
]

# ====================================================================================
# 8. 패키지 로드 시 자동 초기화
# ====================================================================================

# 패키지 로드 시 자동으로 초기화 실행
_initialization_result = initialize_core_package()

if not _initialization_result:
    import warnings

    warnings.warn(
        "Core package initialization failed. Some features may not work correctly.",
        RuntimeWarning,
        stacklevel=2,
    )

# 초기화 성공 메시지 (개발 환경에서만 표시)
import os

if os.getenv("YOKOGAWA_OCR_DEBUG", "false").lower() == "true":
    print(f"✅ YOKOGAWA OCR Core Package v{__version__} initialized successfully")
    print(
        f"   - Abstract Classes: {len([BaseService, BaseModel, BaseProcessor, BaseValidator])}"
    )
    print(
        f"   - Interfaces: {len([DataCollectionInterface, LabelingInterface, AugmentationInterface, ValidationInterface])}"
    )
    print(
        f"   - Exception Classes: {len([cls for cls in globals().values() if isinstance(cls, type) and issubclass(cls, Exception)])}"
    )
