#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 핵심 추상 클래스 및 인터페이스 모듈

이 모듈은 전체 시스템의 기반이 되는 추상 클래스와 인터페이스를 정의합니다.
모든 서비스, 모델, 프로세서, 검증 클래스는 이 모듈의 추상 클래스를 상속받아야 합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import logging
import json
from abc import ABC, abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
    Callable,
    Type,
    TypeVar,
    Generic,
    TYPE_CHECKING,
)
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# 타입 체크 시에만 import (순환 import 방지)
if TYPE_CHECKING:
    from config.settings import ApplicationConfig

from core.exceptions import (
    ApplicationError,
    ValidationError,
    ProcessingError,
    ServiceError,
)

# 타입 변수 정의
T = TypeVar("T")
ModelType = TypeVar("ModelType", bound="BaseModel")
ServiceType = TypeVar("ServiceType", bound="BaseService")


class ServiceStatus(Enum):
    """서비스 상태 열거형"""

    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class ProcessingStatus(Enum):
    """처리 상태 열거형"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ServiceMetrics:
    """서비스 성능 메트릭"""

    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_duration: Optional[timedelta] = None
    items_processed: int = 0
    items_failed: int = 0
    success_rate: float = 0.0

    def calculate_success_rate(self) -> float:
        """성공률 계산"""
        total_items = self.items_processed + self.items_failed
        if total_items == 0:
            return 0.0
        return (self.items_processed / total_items) * 100.0


# ====================================================================================
# 1. 기본 추상 클래스들
# ====================================================================================


class BaseService(ABC):
    """
    모든 서비스 클래스의 기반 추상 클래스

    이 클래스는 모든 서비스에서 공통으로 사용되는 기능을 정의하며,
    의존성 주입 패턴을 지원합니다.
    """

    def __init__(self, config: "ApplicationConfig", logger: logging.Logger):
        """서비스 초기화"""
        self._config = config
        self._logger = logger
        self._service_id = self._generate_service_id()
        self._status = ServiceStatus.UNINITIALIZED
        self._metrics = ServiceMetrics()
        self._error_count = 0
        self._is_initialized = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_start": [],
            "on_stop": [],
            "on_error": [],
            "on_complete": [],
        }
        
        # 서비스 생성 로그
        self._logger.info(
            f"Service {self.__class__.__name__} created with ID: {self._service_id}"
        )

    def _generate_service_id(self) -> str:
        """서비스 고유 ID 생성"""
        import uuid

        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    @property
    def service_id(self) -> str:
        """서비스 ID 반환"""
        return self._service_id

    @property
    def status(self) -> ServiceStatus:
        """서비스 상태 반환"""
        return self._status

    @property
    def config(self) -> "ApplicationConfig":
        """설정 객체 반환"""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """로거 객체 반환"""
        return self._logger

    @property
    def metrics(self) -> ServiceMetrics:
        """서비스 메트릭 반환"""
        return self._metrics

    @abstractmethod
    def initialize(self) -> bool:
        """
        서비스 초기화 - 모든 서비스에서 구현 필수

        Returns:
            bool: 초기화 성공 여부

        Raises:
            ServiceError: 초기화 실패 시
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """
        서비스 정리 - 모든 서비스에서 구현 필수

        Raises:
            ServiceError: 정리 실패 시
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        if not self._is_initialized:
            return False
        
        if self._status != ServiceStatus.RUNNING:
            return False
        
        return True

    def is_initialized(self) -> bool:
        """초기화 상태 확인"""
        return self._is_initialized

    def start(self) -> bool:
        """서비스 시작"""
        try:
            if self._status == ServiceStatus.RUNNING:
                self._logger.warning(f"Service {self.service_id} is already running")
                return True
            
            self._logger.info(f"Starting service {self.service_id}")
            self._status = ServiceStatus.INITIALIZING
            
            # 초기화 수행
            if not self.initialize():
                self._status = ServiceStatus.ERROR
                raise ServiceError(f"Failed to initialize service {self.service_id}")
            
            # 초기화 완료 후 상태 업데이트
            self._is_initialized = True
            self._status = ServiceStatus.RUNNING
            self._metrics.start_time = datetime.now()
            
            # 시작 콜백 실행
            self._execute_callbacks("on_start")
            
            self._logger.info(f"Service {self.service_id} started successfully")
            return True
            
        except Exception as e:
            self._status = ServiceStatus.ERROR
            self._error_count += 1
            self._logger.error(f"Failed to start service {self.service_id}: {str(e)}")
            self._execute_callbacks("on_error", error=e)
            return False

    def stop(self) -> bool:
        """
        서비스 중지

        Returns:
            bool: 중지 성공 여부
        """
        try:
            if self._status == ServiceStatus.STOPPED:
                self._logger.warning(f"Service {self.service_id} is already stopped")
                return True

            self._logger.info(f"Stopping service {self.service_id}")
            self._status = ServiceStatus.STOPPED

            # 정리 수행
            self.cleanup()

            self._metrics.end_time = datetime.now()
            if self._metrics.start_time:
                self._metrics.processing_duration = (
                    self._metrics.end_time - self._metrics.start_time
                )

            # 중지 콜백 실행
            self._execute_callbacks("on_stop")

            self._logger.info(f"Service {self.service_id} stopped successfully")
            return True

        except Exception as e:
            self._status = ServiceStatus.ERROR
            self._error_count += 1
            self._logger.error(f"Failed to stop service {self.service_id}: {str(e)}")
            self._execute_callbacks("on_error", error=e)
            return False

    def pause(self) -> bool:
        """
        서비스 일시 중지

        Returns:
            bool: 일시 중지 성공 여부
        """
        if self._status != ServiceStatus.RUNNING:
            self._logger.warning(
                f"Cannot pause service {self.service_id} - not running"
            )
            return False

        self._status = ServiceStatus.PAUSED
        self._logger.info(f"Service {self.service_id} paused")
        return True

    def resume(self) -> bool:
        """
        서비스 재개

        Returns:
            bool: 재개 성공 여부
        """
        if self._status != ServiceStatus.PAUSED:
            self._logger.warning(
                f"Cannot resume service {self.service_id} - not paused"
            )
            return False

        self._status = ServiceStatus.RUNNING
        self._logger.info(f"Service {self.service_id} resumed")
        return True

    def register_callback(self, event: str, callback: Callable) -> None:
        """
        이벤트 콜백 등록

        Args:
            event: 이벤트 이름 ('on_start', 'on_stop', 'on_error', 'on_complete')
            callback: 콜백 함수
        """
        if event not in self._callbacks:
            raise ValueError(f"Invalid event type: {event}")

        self._callbacks[event].append(callback)
        self._logger.debug(
            f"Callback registered for event '{event}' in service {self.service_id}"
        )

    def _execute_callbacks(self, event_type: str, **kwargs) -> None:
        """콜백 실행"""
        try:
            for callback in self._callbacks.get(event_type, []):
                callback(self, **kwargs)
        except Exception as e:
            self._logger.error(f"Error executing {event_type} callback: {str(e)}")

    def get_status_info(self) -> Dict[str, Any]:
        """
        서비스 상태 정보 반환

        Returns:
            Dict[str, Any]: 상태 정보
        """
        return {
            "service_id": self.service_id,
            "service_name": self.__class__.__name__,
            "status": self._status.value,
            "is_initialized": self._is_initialized,
            "error_count": self._error_count,
            "metrics": {
                "start_time": (
                    self._metrics.start_time.isoformat()
                    if self._metrics.start_time
                    else None
                ),
                "end_time": (
                    self._metrics.end_time.isoformat()
                    if self._metrics.end_time
                    else None
                ),
                "processing_duration": (
                    str(self._metrics.processing_duration)
                    if self._metrics.processing_duration
                    else None
                ),
                "items_processed": self._metrics.items_processed,
                "items_failed": self._metrics.items_failed,
                "success_rate": self._metrics.calculate_success_rate(),
            },
        }


class BaseModel(ABC):
    """
    모든 데이터 모델 클래스의 기반 추상 클래스

    이 클래스는 모든 데이터 모델에서 공통으로 사용되는 기능을 정의합니다.
    """

    def __init__(self):
        """모델 초기화"""
        self._model_id = self._generate_model_id()
        self._created_at = datetime.now()
        self._updated_at = datetime.now()
        self._version = 1
        self._is_valid = False
        self._validation_errors: List[str] = []
        self._metadata: Dict[str, Any] = {}

    def _generate_model_id(self) -> str:
        """모델 고유 ID 생성"""
        import uuid

        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    @property
    def model_id(self) -> str:
        """모델 ID 반환"""
        return self._model_id

    @property
    def created_at(self) -> datetime:
        """생성 시간 반환"""
        return self._created_at

    @property
    def updated_at(self) -> datetime:
        """수정 시간 반환"""
        return self._updated_at

    @property
    def version(self) -> int:
        """버전 반환"""
        return self._version

    @property
    def is_valid(self) -> bool:
        """유효성 상태 반환"""
        return self._is_valid

    @property
    def validation_errors(self) -> List[str]:
        """검증 오류 목록 반환"""
        return self._validation_errors.copy()

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """
        모델을 딕셔너리로 변환 - 모든 모델에서 구현 필수

        Returns:
            Dict[str, Any]: 모델 딕셔너리
        """
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BaseModel":
        """
        딕셔너리에서 모델 인스턴스 생성 - 모든 모델에서 구현 필수

        Args:
            data: 모델 데이터 딕셔너리

        Returns:
            BaseModel: 모델 인스턴스
        """
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        모델 데이터 유효성 검증 - 모든 모델에서 구현 필수

        Returns:
            bool: 유효성 검증 결과
        """
        pass

    def to_json(self) -> str:
        """
        모델을 JSON 문자열로 변환

        Returns:
            str: JSON 문자열
        """
        try:
            return json.dumps(self.to_dict(), default=str, ensure_ascii=False, indent=2)
        except Exception as e:
            raise ProcessingError(f"Failed to convert model to JSON: {str(e)}")

    @classmethod
    def from_json(cls, json_str: str) -> "BaseModel":
        """
        JSON 문자열에서 모델 인스턴스 생성

        Args:
            json_str: JSON 문자열

        Returns:
            BaseModel: 모델 인스턴스
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON format: {str(e)}")
        except Exception as e:
            raise ProcessingError(f"Failed to create model from JSON: {str(e)}")

    def update_metadata(self, key: str, value: Any) -> None:
        """
        메타데이터 업데이트

        Args:
            key: 메타데이터 키
            value: 메타데이터 값
        """
        self._metadata[key] = value
        self._updated_at = datetime.now()
        self._version += 1

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        메타데이터 조회

        Args:
            key: 메타데이터 키
            default: 기본값

        Returns:
            Any: 메타데이터 값
        """
        return self._metadata.get(key, default)

    def clear_validation_errors(self) -> None:
        """검증 오류 초기화"""
        self._validation_errors.clear()

    def add_validation_error(self, error: str) -> None:
        """
        검증 오류 추가

        Args:
            error: 오류 메시지
        """
        self._validation_errors.append(error)
        self._is_valid = False

    def get_basic_info(self) -> Dict[str, Any]:
        """
        기본 정보 반환

        Returns:
            Dict[str, Any]: 기본 정보
        """
        return {
            "model_id": self.model_id,
            "model_type": self.__class__.__name__,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "is_valid": self.is_valid,
            "validation_errors": self.validation_errors,
            "metadata": self._metadata,
        }


class BaseProcessor(ABC):
    """
    모든 프로세서 클래스의 기반 추상 클래스

    이 클래스는 데이터 처리를 담당하는 프로세서들의 공통 인터페이스를 정의합니다.
    """

    def __init__(self, config: "ApplicationConfig", logger: logging.Logger):
        """
        프로세서 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로깅 객체
        """
        self._config = config
        self._logger = logger
        self._processor_id = self._generate_processor_id()
        self._status = ProcessingStatus.PENDING
        self._processed_count = 0
        self._failed_count = 0
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None

    def _generate_processor_id(self) -> str:
        """프로세서 고유 ID 생성"""
        import uuid

        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    @property
    def processor_id(self) -> str:
        """프로세서 ID 반환"""
        return self._processor_id

    @property
    def status(self) -> ProcessingStatus:
        """처리 상태 반환"""
        return self._status

    @property
    def config(self) -> "ApplicationConfig":
        """설정 객체 반환"""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """로거 객체 반환"""
        return self._logger

    @abstractmethod
    def process(self, data: Any) -> Any:
        """
        데이터 처리 - 모든 프로세서에서 구현 필수

        Args:
            data: 처리할 데이터

        Returns:
            Any: 처리된 데이터
        """
        pass

    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """
        입력 데이터 검증 - 모든 프로세서에서 구현 필수

        Args:
            data: 검증할 데이터

        Returns:
            bool: 검증 결과
        """
        pass

    def process_batch(self, data_list: List[Any]) -> List[Any]:
        """
        배치 데이터 처리

        Args:
            data_list: 처리할 데이터 목록

        Returns:
            List[Any]: 처리된 데이터 목록
        """
        results = []
        self._status = ProcessingStatus.PROCESSING
        self._start_time = datetime.now()

        try:
            for data in data_list:
                try:
                    if self.validate_input(data):
                        result = self.process(data)
                        results.append(result)
                        self._processed_count += 1
                    else:
                        self._logger.warning(
                            f"Invalid input data in processor {self.processor_id}"
                        )
                        self._failed_count += 1
                except Exception as e:
                    self._logger.error(
                        f"Error processing data in processor {self.processor_id}: {str(e)}"
                    )
                    self._failed_count += 1

            self._status = ProcessingStatus.COMPLETED
            self._end_time = datetime.now()

        except Exception as e:
            self._status = ProcessingStatus.FAILED
            self._end_time = datetime.now()
            self._logger.error(
                f"Batch processing failed in processor {self.processor_id}: {str(e)}"
            )
            raise ProcessingError(f"Batch processing failed: {str(e)}")

        return results

    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        처리 통계 반환

        Returns:
            Dict[str, Any]: 처리 통계
        """
        duration = None
        if self._start_time and self._end_time:
            duration = (self._end_time - self._start_time).total_seconds()

        return {
            "processor_id": self.processor_id,
            "processor_type": self.__class__.__name__,
            "status": self._status.value,
            "processed_count": self._processed_count,
            "failed_count": self._failed_count,
            "success_rate": (
                self._processed_count
                / (self._processed_count + self._failed_count)
                * 100
                if (self._processed_count + self._failed_count) > 0
                else 0
            ),
            "processing_duration_seconds": duration,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None,
        }


class BaseValidator(ABC):
    """
    모든 검증 클래스의 기반 추상 클래스

    이 클래스는 데이터 검증을 담당하는 검증자들의 공통 인터페이스를 정의합니다.
    """

    def __init__(self, config: "ApplicationConfig", logger: logging.Logger):
        """
        검증자 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로깅 객체
        """
        self._config = config
        self._logger = logger
        self._validator_id = self._generate_validator_id()
        self._validation_rules: Dict[str, Any] = {}
        self._validation_history: List[Dict[str, Any]] = []

    def _generate_validator_id(self) -> str:
        """검증자 고유 ID 생성"""
        import uuid

        return f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"

    @property
    def validator_id(self) -> str:
        """검증자 ID 반환"""
        return self._validator_id

    @property
    def config(self) -> "ApplicationConfig":
        """설정 객체 반환"""
        return self._config

    @property
    def logger(self) -> logging.Logger:
        """로거 객체 반환"""
        return self._logger

    @abstractmethod
    def validate(self, data: Any) -> bool:
        """
        데이터 검증 - 모든 검증자에서 구현 필수

        Args:
            data: 검증할 데이터

        Returns:
            bool: 검증 결과
        """
        pass

    @abstractmethod
    def get_validation_errors(self) -> List[str]:
        """
        검증 오류 목록 반환 - 모든 검증자에서 구현 필수

        Returns:
            List[str]: 오류 목록
        """
        pass

    def add_validation_rule(
        self, rule_name: str, rule_func: Callable[[Any], bool]
    ) -> None:
        """
        검증 규칙 추가

        Args:
            rule_name: 규칙 이름
            rule_func: 규칙 함수
        """
        self._validation_rules[rule_name] = rule_func
        self._logger.debug(
            f"Validation rule '{rule_name}' added to validator {self.validator_id}"
        )

    def remove_validation_rule(self, rule_name: str) -> None:
        """
        검증 규칙 제거

        Args:
            rule_name: 규칙 이름
        """
        if rule_name in self._validation_rules:
            del self._validation_rules[rule_name]
            self._logger.debug(
                f"Validation rule '{rule_name}' removed from validator {self.validator_id}"
            )

    def validate_with_rules(self, data: Any) -> Dict[str, bool]:
        """
        규칙 기반 검증

        Args:
            data: 검증할 데이터

        Returns:
            Dict[str, bool]: 규칙별 검증 결과
        """
        results = {}
        for rule_name, rule_func in self._validation_rules.items():
            try:
                results[rule_name] = rule_func(data)
            except Exception as e:
                self._logger.error(f"Error in validation rule '{rule_name}': {str(e)}")
                results[rule_name] = False

        # 검증 히스토리에 추가
        self._validation_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "data_type": type(data).__name__,
                "results": results,
                "overall_result": all(results.values()),
            }
        )

        return results

    def get_validation_statistics(self) -> Dict[str, Any]:
        """
        검증 통계 반환

        Returns:
            Dict[str, Any]: 검증 통계
        """
        if not self._validation_history:
            return {"total_validations": 0}

        total_validations = len(self._validation_history)
        successful_validations = sum(
            1 for v in self._validation_history if v["overall_result"]
        )

        return {
            "validator_id": self.validator_id,
            "validator_type": self.__class__.__name__,
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "success_rate": (successful_validations / total_validations) * 100,
            "validation_rules_count": len(self._validation_rules),
            "validation_rules": list(self._validation_rules.keys()),
        }


# ====================================================================================
# 2. 서비스 인터페이스 정의
# ====================================================================================


class DataCollectionInterface(ABC):
    """데이터 수집 인터페이스"""

    @abstractmethod
    def collect_files(self, source_path: str) -> List[str]:
        """
        파일 수집

        Args:
            source_path: 원본 파일 경로

        Returns:
            List[str]: 수집된 파일 목록
        """
        pass

    @abstractmethod
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        수집 통계 정보 제공

        Returns:
            Dict[str, Any]: 수집 통계
        """
        pass

    @abstractmethod
    def register_collection_callback(self, callback: Callable) -> None:
        """
        수집 완료 시 콜백 등록

        Args:
            callback: 콜백 함수
        """
        pass


class LabelingInterface(ABC):
    """라벨링 인터페이스"""

    @abstractmethod
    def create_labeling_session(self, file_path: str) -> str:
        """
        라벨링 세션 생성

        Args:
            file_path: 파일 경로

        Returns:
            str: 세션 ID
        """
        pass

    @abstractmethod
    def get_labeling_progress(self) -> Dict[str, float]:
        """
        라벨링 진행 상황 제공

        Returns:
            Dict[str, float]: 진행 상황
        """
        pass

    @abstractmethod
    def export_annotations(self, format_type: str) -> str:
        """
        어노테이션 내보내기

        Args:
            format_type: 내보내기 형식

        Returns:
            str: 내보내기 결과
        """
        pass


class AugmentationInterface(ABC):
    """데이터 증강 인터페이스"""

    @abstractmethod
    def augment_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        데이터셋 증강

        Args:
            dataset: 원본 데이터셋

        Returns:
            List[Dict[str, Any]]: 증강된 데이터셋
        """
        pass

    @abstractmethod
    def get_augmentation_statistics(self) -> Dict[str, Any]:
        """
        증강 통계 정보 제공

        Returns:
            Dict[str, Any]: 증강 통계
        """
        pass

    @abstractmethod
    def configure_augmentation_rules(self, rules: Dict[str, Any]) -> None:
        """
        증강 규칙 설정

        Args:
            rules: 증강 규칙
        """
        pass


class ValidationInterface(ABC):
    """검증 인터페이스"""

    @abstractmethod
    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        데이터셋 검증

        Args:
            dataset: 검증할 데이터셋

        Returns:
            Dict[str, Any]: 검증 결과
        """
        pass

    @abstractmethod
    def get_validation_report(self) -> Dict[str, Any]:
        """
        검증 보고서 생성

        Returns:
            Dict[str, Any]: 검증 보고서
        """
        pass

    @abstractmethod
    def set_validation_criteria(self, criteria: Dict[str, Any]) -> None:
        """
        검증 기준 설정

        Args:
            criteria: 검증 기준
        """
        pass


# ====================================================================================
# 3. 유틸리티 클래스 및 함수
# ====================================================================================


class ServiceRegistry:
    """서비스 레지스트리"""

    def __init__(self):
        self._services: Dict[str, BaseService] = {}
        self._service_types: Dict[str, Type[BaseService]] = {}

    def register_service(
        self, service_name: str, service_instance: BaseService
    ) -> None:
        """
        서비스 등록

        Args:
            service_name: 서비스 이름
            service_instance: 서비스 인스턴스
        """
        self._services[service_name] = service_instance
        self._service_types[service_name] = type(service_instance)

    def get_service(self, service_name: str) -> Optional[BaseService]:
        """
        서비스 조회

        Args:
            service_name: 서비스 이름

        Returns:
            Optional[BaseService]: 서비스 인스턴스
        """
        return self._services.get(service_name)

    def get_all_services(self) -> Dict[str, BaseService]:
        """
        모든 서비스 조회

        Returns:
            Dict[str, BaseService]: 서비스 딕셔너리
        """
        return self._services.copy()

    def get_services_by_status(self, status: ServiceStatus) -> List[BaseService]:
        """
        상태별 서비스 조회

        Args:
            status: 서비스 상태

        Returns:
            List[BaseService]: 서비스 목록
        """
        return [
            service for service in self._services.values() if service.status == status
        ]


def create_service_factory(
    service_class: Type[ServiceType],
) -> Callable[["ApplicationConfig", logging.Logger], ServiceType]:
    """
    서비스 팩토리 생성

    Args:
        service_class: 서비스 클래스

    Returns:
        Callable: 서비스 팩토리 함수
    """

    def factory(config: "ApplicationConfig", logger: logging.Logger) -> ServiceType:
        return service_class(config, logger)

    return factory


def validate_service_interface(
    service: BaseService, interface_class: Type[ABC]
) -> bool:
    """
    서비스 인터페이스 검증

    Args:
        service: 검증할 서비스
        interface_class: 인터페이스 클래스

    Returns:
        bool: 인터페이스 준수 여부
    """
    try:
        return isinstance(service, interface_class)
    except Exception:
        return False


# ====================================================================================
# 4. 모듈 수준 유틸리티
# ====================================================================================


def get_base_class_hierarchy(cls: Type) -> List[Type]:
    """
    클래스의 기본 클래스 계층 반환

    Args:
        cls: 클래스

    Returns:
        List[Type]: 기본 클래스 목록
    """
    hierarchy = []
    for base in cls.__mro__:
        if base in [BaseService, BaseModel, BaseProcessor, BaseValidator]:
            hierarchy.append(base)
    return hierarchy


def is_abstract_implementation_complete(cls: Type, abstract_base: Type[ABC]) -> bool:
    """
    추상 클래스 구현 완료 여부 확인

    Args:
        cls: 확인할 클래스
        abstract_base: 추상 기본 클래스

    Returns:
        bool: 구현 완료 여부
    """
    try:
        # 추상 메서드가 모두 구현되었는지 확인
        abstract_methods = getattr(abstract_base, "__abstractmethods__", set())
        for method_name in abstract_methods:
            if not hasattr(cls, method_name):
                return False
            method = getattr(cls, method_name)
            if getattr(method, "__isabstractmethod__", False):
                return False
        return True
    except Exception:
        return False


if __name__ == "__main__":
    # 기본 클래스 테스트
    print("YOKOGAWA OCR 핵심 클래스 테스트")
    print("=" * 50)

    # 추상 클래스 검증
    abstract_classes = [BaseService, BaseModel, BaseProcessor, BaseValidator]
    for cls in abstract_classes:
        print(f"✅ {cls.__name__} 추상 클래스 정의 완료")

    # 인터페이스 검증
    interfaces = [
        DataCollectionInterface,
        LabelingInterface,
        AugmentationInterface,
        ValidationInterface,
    ]
    for interface in interfaces:
        print(f"✅ {interface.__name__} 인터페이스 정의 완료")

    # 서비스 레지스트리 테스트
    registry = ServiceRegistry()
    print(f"✅ ServiceRegistry 생성 완료")

    print("\n🎯 모든 기본 클래스와 인터페이스가 성공적으로 정의되었습니다!")
