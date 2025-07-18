#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 커스텀 예외 클래스 모듈

이 모듈은 전체 시스템에서 사용되는 커스텀 예외 클래스들을 정의합니다.
모든 예외는 ApplicationError를 기본으로 하는 계층구조를 따릅니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from enum import Enum


class ErrorSeverity(Enum):
    """오류 심각도 레벨 열거형"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCode(Enum):
    """시스템 오류 코드 열거형"""

    # 일반 오류 코드 (1000번대)
    UNKNOWN_ERROR = 1000
    INVALID_PARAMETER = 1001
    INVALID_OPERATION = 1002
    RESOURCE_NOT_FOUND = 1003
    PERMISSION_DENIED = 1004
    TIMEOUT_ERROR = 1005

    # 설정 관련 오류 코드 (2000번대)
    CONFIG_LOAD_ERROR = 2000
    CONFIG_VALIDATION_ERROR = 2001
    CONFIG_MISSING_ERROR = 2002
    CONFIG_FORMAT_ERROR = 2003

    # 서비스 관련 오류 코드 (3000번대)
    SERVICE_INITIALIZATION_ERROR = 3000
    SERVICE_STARTUP_ERROR = 3001
    SERVICE_SHUTDOWN_ERROR = 3002
    SERVICE_COMMUNICATION_ERROR = 3003
    SERVICE_DEPENDENCY_ERROR = 3004

    # 데이터 수집 관련 오류 코드 (4000번대)
    DATA_COLLECTION_ERROR = 4000
    FILE_NOT_FOUND_ERROR = 4001
    FILE_ACCESS_ERROR = 4002
    FILE_FORMAT_ERROR = 4003
    FILE_SIZE_ERROR = 4004
    METADATA_EXTRACTION_ERROR = 4005

    # 라벨링 관련 오류 코드 (5000번대)
    LABELING_SESSION_ERROR = 5000
    ANNOTATION_CREATION_ERROR = 5001
    ANNOTATION_VALIDATION_ERROR = 5002
    ANNOTATION_SAVE_ERROR = 5003
    TEMPLATE_LOAD_ERROR = 5004

    # 데이터 증강 관련 오류 코드 (6000번대)
    AUGMENTATION_ERROR = 6000
    IMAGE_PROCESSING_ERROR = 6001
    TRANSFORMATION_ERROR = 6002
    AUGMENTATION_RULE_ERROR = 6003

    # 검증 관련 오류 코드 (7000번대)
    VALIDATION_ERROR = 7000
    DATA_INTEGRITY_ERROR = 7001
    QUALITY_CHECK_ERROR = 7002
    CONSISTENCY_CHECK_ERROR = 7003

    # 처리 관련 오류 코드 (8000번대)
    PROCESSING_ERROR = 8000
    BATCH_PROCESSING_ERROR = 8001
    PARALLEL_PROCESSING_ERROR = 8002
    MEMORY_ERROR = 8003

    # 파일 처리 관련 오류 코드 (9000번대)
    FILE_PROCESSING_ERROR = 9000
    PDF_PROCESSING_ERROR = 9001
    IMAGE_CONVERSION_ERROR = 9002
    FILE_COMPRESSION_ERROR = 9003


# ====================================================================================
# 1. 기본 예외 클래스
# ====================================================================================


class ApplicationError(Exception):
    """
    모든 애플리케이션 예외의 기본 클래스

    이 클래스는 모든 커스텀 예외의 기본이 되며,
    오류 추적 및 로깅을 위한 공통 기능을 제공합니다.
    """

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        """
        ApplicationError 초기화

        Args:
            message: 오류 메시지
            error_code: 오류 코드
            severity: 오류 심각도
            context: 추가 컨텍스트 정보
            original_exception: 원본 예외 (체인된 예외인 경우)
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.severity = severity
        self.context = context or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()
        self.stack_trace = traceback.format_exc()

        # 오류 ID 생성
        self.error_id = self._generate_error_id()

        # 자동 로깅
        self._log_error()

    def _generate_error_id(self) -> str:
        """오류 고유 ID 생성"""
        import uuid

        return f"ERR_{self.error_code.value}_{uuid.uuid4().hex[:8]}"

    def _log_error(self) -> None:
        """오류 자동 로깅"""
        logger = logging.getLogger(self.__class__.__name__)

        log_message = f"[{self.error_id}] {self.message}"

        if self.context:
            log_message += f" | Context: {self.context}"

        if self.original_exception:
            log_message += f" | Original: {str(self.original_exception)}"

        # 심각도에 따른 로그 레벨 결정
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def to_dict(self) -> Dict[str, Any]:
        """
        예외 정보를 딕셔너리로 변환

        Returns:
            Dict[str, Any]: 예외 정보 딕셔너리
        """
        return {
            "error_id": self.error_id,
            "error_code": self.error_code.value,
            "error_name": self.error_code.name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "timestamp": self.timestamp.isoformat(),
            "exception_type": self.__class__.__name__,
            "original_exception": (
                str(self.original_exception) if self.original_exception else None
            ),
        }

    def add_context(self, key: str, value: Any) -> None:
        """
        컨텍스트 정보 추가

        Args:
            key: 컨텍스트 키
            value: 컨텍스트 값
        """
        self.context[key] = value

    def get_context(self, key: str, default: Any = None) -> Any:
        """
        컨텍스트 정보 조회

        Args:
            key: 컨텍스트 키
            default: 기본값

        Returns:
            Any: 컨텍스트 값
        """
        return self.context.get(key, default)

    def is_critical(self) -> bool:
        """치명적 오류 여부 확인"""
        return self.severity == ErrorSeverity.CRITICAL

    def is_retryable(self) -> bool:
        """재시도 가능한 오류 여부 확인"""
        # 기본적으로 일시적 오류는 재시도 가능
        retryable_codes = [
            ErrorCode.TIMEOUT_ERROR,
            ErrorCode.SERVICE_COMMUNICATION_ERROR,
            ErrorCode.MEMORY_ERROR,
        ]
        return self.error_code in retryable_codes

    def __str__(self) -> str:
        """문자열 표현"""
        return f"[{self.error_code.name}] {self.message}"

    def __repr__(self) -> str:
        """개발자용 표현"""
        return (
            f"{self.__class__.__name__}("
            f"message='{self.message}', "
            f"error_code={self.error_code.name}, "
            f"severity={self.severity.name}, "
            f"error_id='{self.error_id}'"
            f")"
        )


# ====================================================================================
# 2. 설정 관련 예외 클래스
# ====================================================================================


class ConfigurationError(ApplicationError):
    """
    설정 관련 오류 예외 클래스

    설정 파일 로드, 검증, 파싱 등의 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        config_file: Optional[str] = None,
        config_section: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.CONFIG_LOAD_ERROR,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        original_exception: Optional[Exception] = None,
    ):
        """
        ConfigurationError 초기화

        Args:
            message: 오류 메시지
            config_file: 설정 파일 경로
            config_section: 설정 섹션
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if config_file:
            context["config_file"] = config_file
        if config_section:
            context["config_section"] = config_section

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.config_file = config_file
        self.config_section = config_section


class ConfigurationValidationError(ConfigurationError):
    """설정 검증 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        validation_rules: Optional[List[str]] = None,
        failed_validations: Optional[List[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if validation_rules:
            context["validation_rules"] = validation_rules
        if failed_validations:
            context["failed_validations"] = failed_validations

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.CONFIG_VALIDATION_ERROR

        super().__init__(message, **kwargs)

        self.validation_rules = validation_rules or []
        self.failed_validations = failed_validations or []


# ====================================================================================
# 3. 서비스 관련 예외 클래스
# ====================================================================================


class ServiceError(ApplicationError):
    """
    서비스 관련 오류 예외 클래스

    서비스 초기화, 시작, 중지, 통신 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        service_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.SERVICE_INITIALIZATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.HIGH,
        original_exception: Optional[Exception] = None,
    ):
        """
        ServiceError 초기화

        Args:
            message: 오류 메시지
            service_name: 서비스 이름
            service_id: 서비스 ID
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if service_name:
            context["service_name"] = service_name
        if service_id:
            context["service_id"] = service_id

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.service_name = service_name
        self.service_id = service_id


class ServiceDependencyError(ServiceError):
    """서비스 의존성 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        missing_dependencies: Optional[List[str]] = None,
        circular_dependencies: Optional[List[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if missing_dependencies:
            context["missing_dependencies"] = missing_dependencies
        if circular_dependencies:
            context["circular_dependencies"] = circular_dependencies

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.SERVICE_DEPENDENCY_ERROR

        super().__init__(message, **kwargs)

        self.missing_dependencies = missing_dependencies or []
        self.circular_dependencies = circular_dependencies or []


# ====================================================================================
# 4. 데이터 수집 관련 예외 클래스
# ====================================================================================


class DataCollectionError(ApplicationError):
    """
    데이터 수집 관련 오류 예외 클래스

    파일 수집, 메타데이터 추출, 중복 탐지 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        source_path: Optional[str] = None,
        file_count: Optional[int] = None,
        error_code: ErrorCode = ErrorCode.DATA_COLLECTION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        DataCollectionError 초기화

        Args:
            message: 오류 메시지
            source_path: 소스 경로
            file_count: 파일 개수
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if source_path:
            context["source_path"] = source_path
        if file_count is not None:
            context["file_count"] = file_count

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.source_path = source_path
        self.file_count = file_count


class FileAccessError(DataCollectionError):
    """파일 접근 오류 예외 클래스"""

    def __init__(
        self, message: str, file_path: str, access_type: str = "read", **kwargs
    ):
        context = kwargs.get("context", {})
        context["file_path"] = file_path
        context["access_type"] = access_type

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.FILE_ACCESS_ERROR

        super().__init__(message, **kwargs)

        self.file_path = file_path
        self.access_type = access_type


class FileFormatError(DataCollectionError):
    """파일 형식 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        file_path: str,
        expected_format: str,
        actual_format: Optional[str] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        context["file_path"] = file_path
        context["expected_format"] = expected_format
        if actual_format:
            context["actual_format"] = actual_format

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.FILE_FORMAT_ERROR

        super().__init__(message, **kwargs)

        self.file_path = file_path
        self.expected_format = expected_format
        self.actual_format = actual_format


# ====================================================================================
# 5. 라벨링 관련 예외 클래스
# ====================================================================================


class LabelingError(ApplicationError):
    """
    라벨링 관련 오류 예외 클래스

    어노테이션 생성, 세션 관리, 품질 제어 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        document_id: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.LABELING_SESSION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        LabelingError 초기화

        Args:
            message: 오류 메시지
            session_id: 세션 ID
            document_id: 문서 ID
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if session_id:
            context["session_id"] = session_id
        if document_id:
            context["document_id"] = document_id

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.session_id = session_id
        self.document_id = document_id


class AnnotationValidationError(LabelingError):
    """어노테이션 검증 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        annotation_id: Optional[str] = None,
        validation_failures: Optional[List[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if annotation_id:
            context["annotation_id"] = annotation_id
        if validation_failures:
            context["validation_failures"] = validation_failures

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.ANNOTATION_VALIDATION_ERROR

        super().__init__(message, **kwargs)

        self.annotation_id = annotation_id
        self.validation_failures = validation_failures or []


# ====================================================================================
# 6. 데이터 증강 관련 예외 클래스
# ====================================================================================


class AugmentationError(ApplicationError):
    """
    데이터 증강 관련 오류 예외 클래스

    이미지 변환, 기하학적 변형, 색상 조정 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        augmentation_type: Optional[str] = None,
        input_data_type: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.AUGMENTATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        AugmentationError 초기화

        Args:
            message: 오류 메시지
            augmentation_type: 증강 유형
            input_data_type: 입력 데이터 유형
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if augmentation_type:
            context["augmentation_type"] = augmentation_type
        if input_data_type:
            context["input_data_type"] = input_data_type

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.augmentation_type = augmentation_type
        self.input_data_type = input_data_type


class ImageProcessingError(AugmentationError):
    """이미지 처리 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        image_path: Optional[str] = None,
        image_dimensions: Optional[tuple] = None,
        processing_operation: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if image_path:
            context["image_path"] = image_path
        if image_dimensions:
            context["image_dimensions"] = image_dimensions
        if processing_operation:
            context["processing_operation"] = processing_operation

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.IMAGE_PROCESSING_ERROR
        kwargs["augmentation_type"] = "image_processing"

        super().__init__(message, **kwargs)

        self.image_path = image_path
        self.image_dimensions = image_dimensions
        self.processing_operation = processing_operation


# ====================================================================================
# 7. 검증 관련 예외 클래스
# ====================================================================================


class ValidationError(ApplicationError):
    """
    검증 관련 오류 예외 클래스

    데이터 품질 검증, 무결성 확인, 일관성 검사 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        validation_type: Optional[str] = None,
        failed_checks: Optional[List[str]] = None,
        error_code: ErrorCode = ErrorCode.VALIDATION_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        ValidationError 초기화

        Args:
            message: 오류 메시지
            validation_type: 검증 유형
            failed_checks: 실패한 검사 목록
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if validation_type:
            context["validation_type"] = validation_type
        if failed_checks:
            context["failed_checks"] = failed_checks

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.validation_type = validation_type
        self.failed_checks = failed_checks or []


class DataIntegrityError(ValidationError):
    """데이터 무결성 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        data_type: Optional[str] = None,
        integrity_checks: Optional[List[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if data_type:
            context["data_type"] = data_type
        if integrity_checks:
            context["integrity_checks"] = integrity_checks

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.DATA_INTEGRITY_ERROR
        kwargs["validation_type"] = "data_integrity"

        super().__init__(message, **kwargs)

        self.data_type = data_type
        self.integrity_checks = integrity_checks or []


# ====================================================================================
# 8. 처리 관련 예외 클래스
# ====================================================================================


class ProcessingError(ApplicationError):
    """
    처리 관련 오류 예외 클래스

    배치 처리, 병렬 처리, 메모리 부족 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        processor_id: Optional[str] = None,
        processing_stage: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.PROCESSING_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        ProcessingError 초기화

        Args:
            message: 오류 메시지
            processor_id: 프로세서 ID
            processing_stage: 처리 단계
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if processor_id:
            context["processor_id"] = processor_id
        if processing_stage:
            context["processing_stage"] = processing_stage

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.processor_id = processor_id
        self.processing_stage = processing_stage


class BatchProcessingError(ProcessingError):
    """배치 처리 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        batch_id: Optional[str] = None,
        batch_size: Optional[int] = None,
        failed_items: Optional[List[str]] = None,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if batch_id:
            context["batch_id"] = batch_id
        if batch_size is not None:
            context["batch_size"] = batch_size
        if failed_items:
            context["failed_items"] = failed_items

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.BATCH_PROCESSING_ERROR
        kwargs["processing_stage"] = "batch_processing"

        super().__init__(message, **kwargs)

        self.batch_id = batch_id
        self.batch_size = batch_size
        self.failed_items = failed_items or []


# ====================================================================================
# 9. 파일 처리 관련 예외 클래스
# ====================================================================================


class FileProcessingError(ApplicationError):
    """
    파일 처리 관련 오류 예외 클래스

    PDF 처리, 이미지 변환, 파일 압축 오류 시 발생합니다.
    """

    def __init__(
        self,
        message: str,
        file_path: Optional[str] = None,
        file_type: Optional[str] = None,
        operation: Optional[str] = None,
        error_code: ErrorCode = ErrorCode.FILE_PROCESSING_ERROR,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        original_exception: Optional[Exception] = None,
    ):
        """
        FileProcessingError 초기화

        Args:
            message: 오류 메시지
            file_path: 파일 경로
            file_type: 파일 타입
            operation: 수행 중인 작업
            error_code: 오류 코드
            severity: 오류 심각도
            original_exception: 원본 예외
        """
        context = {}
        if file_path:
            context["file_path"] = file_path
        if file_type:
            context["file_type"] = file_type
        if operation:
            context["operation"] = operation

        super().__init__(
            message=message,
            error_code=error_code,
            severity=severity,
            context=context,
            original_exception=original_exception,
        )

        self.file_path = file_path
        self.file_type = file_type
        self.operation = operation


class PDFProcessingError(FileProcessingError):
    """PDF 처리 오류 예외 클래스"""

    def __init__(
        self,
        message: str,
        page_number: Optional[int] = None,
        pdf_encrypted: bool = False,
        **kwargs,
    ):
        context = kwargs.get("context", {})
        if page_number is not None:
            context["page_number"] = page_number
        context["pdf_encrypted"] = pdf_encrypted

        kwargs["context"] = context
        kwargs["error_code"] = ErrorCode.PDF_PROCESSING_ERROR
        kwargs["file_type"] = "pdf"

        super().__init__(message, **kwargs)

        self.page_number = page_number
        self.pdf_encrypted = pdf_encrypted


# ====================================================================================
# 10. 유틸리티 함수들
# ====================================================================================


def create_error_from_exception(
    exception: Exception,
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    additional_context: Optional[Dict[str, Any]] = None,
) -> ApplicationError:
    """
    기존 예외에서 ApplicationError 생성

    Args:
        exception: 원본 예외
        error_code: 오류 코드
        severity: 오류 심각도
        additional_context: 추가 컨텍스트

    Returns:
        ApplicationError: 생성된 ApplicationError 인스턴스
    """
    message = str(exception)
    context = additional_context or {}
    context["original_exception_type"] = type(exception).__name__

    return ApplicationError(
        message=message,
        error_code=error_code,
        severity=severity,
        context=context,
        original_exception=exception,
    )


def chain_exceptions(
    new_exception: ApplicationError, original_exception: Exception
) -> ApplicationError:
    """
    예외 체인 생성

    Args:
        new_exception: 새로운 예외
        original_exception: 원본 예외

    Returns:
        ApplicationError: 체인된 예외
    """
    new_exception.original_exception = original_exception
    new_exception.add_context("chained_from", type(original_exception).__name__)
    return new_exception


def get_error_summary(errors: List[ApplicationError]) -> Dict[str, Any]:
    """
    오류 목록 요약 정보 생성

    Args:
        errors: 오류 목록

    Returns:
        Dict[str, Any]: 요약 정보
    """
    if not errors:
        return {"total_errors": 0}

    # 오류 코드별 집계
    error_codes = {}
    for error in errors:
        code = error.error_code.name
        error_codes[code] = error_codes.get(code, 0) + 1

    # 심각도별 집계
    severities = {}
    for error in errors:
        severity = error.severity.name
        severities[severity] = severities.get(severity, 0) + 1

    # 시간 범위 계산
    timestamps = [error.timestamp for error in errors]
    earliest = min(timestamps)
    latest = max(timestamps)

    return {
        "total_errors": len(errors),
        "error_codes": error_codes,
        "severities": severities,
        "time_range": {
            "earliest": earliest.isoformat(),
            "latest": latest.isoformat(),
            "duration_seconds": (latest - earliest).total_seconds(),
        },
        "critical_errors": len(
            [e for e in errors if e.severity == ErrorSeverity.CRITICAL]
        ),
        "retryable_errors": len([e for e in errors if e.is_retryable()]),
    }


def filter_errors_by_severity(
    errors: List[ApplicationError], min_severity: ErrorSeverity = ErrorSeverity.MEDIUM
) -> List[ApplicationError]:
    """
    심각도에 따른 오류 필터링

    Args:
        errors: 오류 목록
        min_severity: 최소 심각도

    Returns:
        List[ApplicationError]: 필터링된 오류 목록
    """
    severity_order = {
        ErrorSeverity.LOW: 1,
        ErrorSeverity.MEDIUM: 2,
        ErrorSeverity.HIGH: 3,
        ErrorSeverity.CRITICAL: 4,
    }

    min_level = severity_order[min_severity]
    return [error for error in errors if severity_order[error.severity] >= min_level]


def group_errors_by_code(
    errors: List[ApplicationError],
) -> Dict[str, List[ApplicationError]]:
    """
    오류 코드별 그룹화

    Args:
        errors: 오류 목록

    Returns:
        Dict[str, List[ApplicationError]]: 코드별 그룹화된 오류
    """
    grouped = {}
    for error in errors:
        code = error.error_code.name
        if code not in grouped:
            grouped[code] = []
        grouped[code].append(error)

    return grouped


# ====================================================================================
# 11. 예외 처리 데코레이터
# ====================================================================================


def handle_exceptions(
    error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    reraise: bool = True,
    default_return: Any = None,
):
    """
    예외 처리 데코레이터

    Args:
        error_code: 기본 오류 코드
        severity: 기본 심각도
        reraise: 예외 재발생 여부
        default_return: 기본 반환값
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ApplicationError:
                # 이미 ApplicationError인 경우 그대로 전파
                if reraise:
                    raise
                return default_return
            except Exception as e:
                # 일반 예외를 ApplicationError로 변환
                app_error = create_error_from_exception(e, error_code, severity)
                if reraise:
                    raise app_error
                return default_return

        return wrapper

    return decorator


# ====================================================================================
# 12. 런타임 예외 검증
# ====================================================================================


def validate_error_hierarchy() -> bool:
    """
    예외 계층구조 검증

    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 모든 커스텀 예외가 ApplicationError를 상속받는지 확인
        custom_exceptions = [
            ConfigurationError,
            ServiceError,
            DataCollectionError,
            LabelingError,
            AugmentationError,
            ValidationError,
            ProcessingError,
            FileProcessingError,
        ]

        for exc_class in custom_exceptions:
            if not issubclass(exc_class, ApplicationError):
                return False

        # 특화된 예외들이 올바른 부모 클래스를 상속받는지 확인
        specialized_exceptions = [
            (ConfigurationValidationError, ConfigurationError),
            (ServiceDependencyError, ServiceError),
            (FileAccessError, DataCollectionError),
            (FileFormatError, DataCollectionError),
            (AnnotationValidationError, LabelingError),
            (ImageProcessingError, AugmentationError),
            (DataIntegrityError, ValidationError),
            (BatchProcessingError, ProcessingError),
            (PDFProcessingError, FileProcessingError),
        ]

        for child_class, parent_class in specialized_exceptions:
            if not issubclass(child_class, parent_class):
                return False

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # 예외 계층구조 테스트
    print("YOKOGAWA OCR 예외 클래스 테스트")
    print("=" * 50)

    # 계층구조 검증
    if validate_error_hierarchy():
        print("✅ 예외 계층구조 검증 완료")
    else:
        print("❌ 예외 계층구조 검증 실패")

    # 오류 코드 테스트
    print(f"✅ 총 {len(ErrorCode)} 개의 오류 코드 정의됨")

    # 심각도 레벨 테스트
    print(f"✅ 총 {len(ErrorSeverity)} 개의 심각도 레벨 정의됨")

    # 예외 생성 테스트
    try:
        test_error = ApplicationError(
            message="테스트 오류",
            error_code=ErrorCode.UNKNOWN_ERROR,
            severity=ErrorSeverity.LOW,
            context={"test": True},
        )
        print(f"✅ 예외 생성 테스트 성공: {test_error.error_id}")
    except Exception as e:
        print(f"❌ 예외 생성 테스트 실패: {e}")

    print("\n🎯 모든 예외 클래스가 성공적으로 정의되었습니다!")
