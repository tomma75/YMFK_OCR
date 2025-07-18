#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 서비스 패키지 초기화 모듈

이 모듈은 모든 서비스 클래스들을 노출하고, 서비스 팩토리와 레지스트리를 제공합니다.
의존성 주입 패턴을 통해 서비스 간 결합도를 낮추고 테스트 가능성을 높입니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import logging
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, Callable
from abc import ABC, abstractmethod
from datetime import datetime
import threading

# 메인 서비스 클래스 임포트
from .data_collection_service import (
    DataCollectionService,
    FileCollector,
    MetadataExtractor,
    DuplicateDetector,
    create_data_collection_service,
)

from .labeling_service import (
    LabelingService,
    AnnotationManager,
    QualityController,
    LabelingSessionManager,
    create_labeling_service,
)

from .augmentation_service import (
    AugmentationService,
    ImageAugmenter,
    GeometricTransformer,
    ColorAdjuster,
    NoiseGenerator,
    create_augmentation_service,
)

from .validation_service import (
    ValidationService,
    DataQualityValidator,
    ConsistencyChecker,
    StatisticsGenerator,
    create_validation_service,
)

# 타입 변수 정의
T = TypeVar("T")
ServiceType = TypeVar("ServiceType")

# 서비스 타입 정의
SERVICE_TYPES = {
    "data_collection": DataCollectionService,
    "labeling": LabelingService,
    "augmentation": AugmentationService,
    "validation": ValidationService,
}

# 헬퍼 클래스 타입 정의
HELPER_CLASSES = {
    # 데이터 수집 헬퍼
    "file_collector": FileCollector,
    "metadata_extractor": MetadataExtractor,
    "duplicate_detector": DuplicateDetector,
    # 라벨링 헬퍼
    "annotation_manager": AnnotationManager,
    "quality_controller": QualityController,
    "labeling_session_manager": LabelingSessionManager,
    # 데이터 증강 헬퍼
    "image_augmenter": ImageAugmenter,
    "geometric_transformer": GeometricTransformer,
    "color_adjuster": ColorAdjuster,
    "noise_generator": NoiseGenerator,
    # 검증 헬퍼
    "data_quality_validator": DataQualityValidator,
    "consistency_checker": ConsistencyChecker,
    "statistics_generator": StatisticsGenerator,
}


class ServiceRegistry:
    """
    서비스 레지스트리 클래스

    애플리케이션 전체에서 사용되는 서비스 인스턴스들을 관리합니다.
    싱글톤 패턴을 적용하여 서비스 인스턴스의 일관성을 보장합니다.
    """

    _instance: Optional["ServiceRegistry"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ServiceRegistry":
        """
        싱글톤 패턴 구현

        Returns:
            ServiceRegistry: 서비스 레지스트리 인스턴스
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """
        ServiceRegistry 초기화
        """
        if not hasattr(self, "_initialized"):
            self._services: Dict[str, Any] = {}
            self._service_factories: Dict[str, Callable] = {}
            self._service_metadata: Dict[str, Dict[str, Any]] = {}
            self._initialization_order: List[str] = []
            self._logger = logging.getLogger(__name__)
            self._initialized = True

    def register_service(
        self,
        service_name: str,
        service_instance: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        서비스 인스턴스 등록

        Args:
            service_name: 서비스 이름
            service_instance: 서비스 인스턴스
            metadata: 서비스 메타데이터
        """
        with self._lock:
            self._services[service_name] = service_instance
            self._service_metadata[service_name] = metadata or {}
            self._service_metadata[service_name]["registered_at"] = datetime.now()

            if service_name not in self._initialization_order:
                self._initialization_order.append(service_name)

            self._logger.info(f"Service registered: {service_name}")

    def register_service_factory(
        self,
        service_name: str,
        factory_function: Callable,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        서비스 팩토리 함수 등록

        Args:
            service_name: 서비스 이름
            factory_function: 팩토리 함수
            metadata: 서비스 메타데이터
        """
        with self._lock:
            self._service_factories[service_name] = factory_function
            self._service_metadata[service_name] = metadata or {}
            self._service_metadata[service_name][
                "factory_registered_at"
            ] = datetime.now()

            self._logger.info(f"Service factory registered: {service_name}")

    def get_service(self, service_name: str) -> Any:
        """
        서비스 인스턴스 조회

        Args:
            service_name: 서비스 이름

        Returns:
            Any: 서비스 인스턴스

        Raises:
            ValueError: 서비스가 등록되지 않은 경우
        """
        with self._lock:
            if service_name in self._services:
                return self._services[service_name]
            elif service_name in self._service_factories:
                # 팩토리 함수를 통해 서비스 생성
                factory = self._service_factories[service_name]
                service_instance = factory()
                self._services[service_name] = service_instance
                return service_instance
            else:
                raise ValueError(f"Service not found: {service_name}")

    def has_service(self, service_name: str) -> bool:
        """
        서비스 등록 여부 확인

        Args:
            service_name: 서비스 이름

        Returns:
            bool: 서비스 등록 여부
        """
        with self._lock:
            return (
                service_name in self._services
                or service_name in self._service_factories
            )

    def get_service_metadata(self, service_name: str) -> Dict[str, Any]:
        """
        서비스 메타데이터 조회

        Args:
            service_name: 서비스 이름

        Returns:
            Dict[str, Any]: 서비스 메타데이터
        """
        with self._lock:
            return self._service_metadata.get(service_name, {})

    def get_registered_services(self) -> List[str]:
        """
        등록된 서비스 목록 조회

        Returns:
            List[str]: 등록된 서비스 이름 목록
        """
        with self._lock:
            return list(self._services.keys()) + list(self._service_factories.keys())

    def clear_services(self) -> None:
        """
        모든 서비스 등록 해제
        """
        with self._lock:
            self._services.clear()
            self._service_factories.clear()
            self._service_metadata.clear()
            self._initialization_order.clear()
            self._logger.info("All services cleared")

    def get_service_statistics(self) -> Dict[str, Any]:
        """
        서비스 레지스트리 통계 정보 반환

        Returns:
            Dict[str, Any]: 통계 정보
        """
        with self._lock:
            return {
                "total_services": len(self._services),
                "total_factories": len(self._service_factories),
                "initialization_order": self._initialization_order.copy(),
                "service_names": list(self._services.keys()),
                "factory_names": list(self._service_factories.keys()),
            }


class ServiceFactory:
    """
    서비스 팩토리 클래스

    서비스 인스턴스 생성을 위한 팩토리 메서드들을 제공합니다.
    의존성 주입 패턴을 통해 서비스 간 결합도를 낮춥니다.
    """

    def __init__(self, service_registry: ServiceRegistry):
        """
        ServiceFactory 초기화

        Args:
            service_registry: 서비스 레지스트리
        """
        self.service_registry = service_registry
        self.logger = logging.getLogger(__name__)

    def create_service(self, service_type: str, config: Any, **kwargs) -> Any:
        """
        서비스 인스턴스 생성

        Args:
            service_type: 서비스 타입
            config: 설정 객체
            **kwargs: 추가 매개변수

        Returns:
            Any: 생성된 서비스 인스턴스

        Raises:
            ValueError: 지원하지 않는 서비스 타입인 경우
        """
        try:
            if service_type == "data_collection":
                return create_data_collection_service(config)
            elif service_type == "labeling":
                return create_labeling_service(config)
            elif service_type == "augmentation":
                return create_augmentation_service(config)
            elif service_type == "validation":
                return create_validation_service(config)
            else:
                raise ValueError(f"Unsupported service type: {service_type}")

        except Exception as e:
            self.logger.error(f"Failed to create service {service_type}: {str(e)}")
            raise

    def create_all_services(self, config: Any) -> Dict[str, Any]:
        """
        모든 서비스 인스턴스 생성

        Args:
            config: 설정 객체

        Returns:
            Dict[str, Any]: 생성된 서비스 인스턴스들
        """
        services = {}

        # 서비스 생성 순서 (의존성 고려)
        service_creation_order = [
            "data_collection",
            "labeling",
            "augmentation",
            "validation",
        ]

        for service_type in service_creation_order:
            try:
                service_instance = self.create_service(service_type, config)
                services[service_type] = service_instance

                # 서비스 레지스트리에 등록
                self.service_registry.register_service(
                    service_type,
                    service_instance,
                    {
                        "service_type": service_type,
                        "created_at": datetime.now(),
                        "config_version": getattr(config, "version", "unknown"),
                    },
                )

                self.logger.info(f"Service created and registered: {service_type}")

            except Exception as e:
                self.logger.error(f"Failed to create service {service_type}: {str(e)}")
                raise

        return services

    def create_helper_instance(self, helper_type: str, config: Any, **kwargs) -> Any:
        """
        헬퍼 클래스 인스턴스 생성

        Args:
            helper_type: 헬퍼 클래스 타입
            config: 설정 객체
            **kwargs: 추가 매개변수

        Returns:
            Any: 생성된 헬퍼 인스턴스

        Raises:
            ValueError: 지원하지 않는 헬퍼 타입인 경우
        """
        if helper_type not in HELPER_CLASSES:
            raise ValueError(f"Unsupported helper type: {helper_type}")

        helper_class = HELPER_CLASSES[helper_type]

        try:
            return helper_class(config, **kwargs)
        except Exception as e:
            self.logger.error(f"Failed to create helper {helper_type}: {str(e)}")
            raise


# 전역 서비스 레지스트리 및 팩토리 인스턴스
_service_registry = ServiceRegistry()
_service_factory = ServiceFactory(_service_registry)


# 편의성 함수들
def get_service_registry() -> ServiceRegistry:
    """
    전역 서비스 레지스트리 조회

    Returns:
        ServiceRegistry: 서비스 레지스트리 인스턴스
    """
    return _service_registry


def get_service_factory() -> ServiceFactory:
    """
    전역 서비스 팩토리 조회

    Returns:
        ServiceFactory: 서비스 팩토리 인스턴스
    """
    return _service_factory


def register_service(
    service_name: str, service_instance: Any, metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    서비스 등록 편의 함수

    Args:
        service_name: 서비스 이름
        service_instance: 서비스 인스턴스
        metadata: 서비스 메타데이터
    """
    _service_registry.register_service(service_name, service_instance, metadata)


def get_service(service_name: str) -> Any:
    """
    서비스 조회 편의 함수

    Args:
        service_name: 서비스 이름

    Returns:
        Any: 서비스 인스턴스
    """
    return _service_registry.get_service(service_name)


def create_service(service_type: str, config: Any, **kwargs) -> Any:
    """
    서비스 생성 편의 함수

    Args:
        service_type: 서비스 타입
        config: 설정 객체
        **kwargs: 추가 매개변수

    Returns:
        Any: 생성된 서비스 인스턴스
    """
    return _service_factory.create_service(service_type, config, **kwargs)


def initialize_all_services(config: Any) -> Dict[str, Any]:
    """
    모든 서비스 초기화 편의 함수

    Args:
        config: 설정 객체

    Returns:
        Dict[str, Any]: 초기화된 서비스 인스턴스들
    """
    return _service_factory.create_all_services(config)


def get_service_statistics() -> Dict[str, Any]:
    """
    서비스 통계 조회 편의 함수

    Returns:
        Dict[str, Any]: 서비스 통계 정보
    """
    return _service_registry.get_service_statistics()


def cleanup_services() -> None:
    """
    모든 서비스 정리 편의 함수
    """
    registry = get_service_registry()
    services = registry.get_registered_services()

    for service_name in services:
        try:
            service = registry.get_service(service_name)
            if hasattr(service, "cleanup"):
                service.cleanup()
        except Exception as e:
            logging.error(f"Error cleaning up service {service_name}: {str(e)}")

    registry.clear_services()


# 공개 API 정의
__all__ = [
    # 메인 서비스 클래스
    "DataCollectionService",
    "LabelingService",
    "AugmentationService",
    "ValidationService",
    # 헬퍼 클래스
    "FileCollector",
    "MetadataExtractor",
    "DuplicateDetector",
    "AnnotationManager",
    "QualityController",
    "LabelingSessionManager",
    "ImageAugmenter",
    "GeometricTransformer",
    "ColorAdjuster",
    "NoiseGenerator",
    "DataQualityValidator",
    "ConsistencyChecker",
    "StatisticsGenerator",
    # 팩토리 및 레지스트리
    "ServiceRegistry",
    "ServiceFactory",
    # 편의 함수
    "get_service_registry",
    "get_service_factory",
    "register_service",
    "get_service",
    "create_service",
    "initialize_all_services",
    "get_service_statistics",
    "cleanup_services",
    # 팩토리 함수
    "create_data_collection_service",
    "create_labeling_service",
    "create_augmentation_service",
    "create_validation_service",
    # 상수
    "SERVICE_TYPES",
    "HELPER_CLASSES",
]


# 모듈 수준 초기화
def _initialize_module():
    """모듈 수준 초기화 함수"""
    logger = logging.getLogger(__name__)

    # 기본 팩토리 함수들 등록
    registry = get_service_registry()

    # 서비스 팩토리 함수들을 레지스트리에 등록
    factory_functions = {
        "data_collection": lambda: create_data_collection_service,
        "labeling": lambda: create_labeling_service,
        "augmentation": lambda: create_augmentation_service,
        "validation": lambda: create_validation_service,
    }

    for service_name, factory_func in factory_functions.items():
        registry.register_service_factory(
            service_name,
            factory_func,
            {
                "service_type": service_name,
                "factory_type": "module_level",
                "registered_at": datetime.now(),
            },
        )

    logger.info("Services module initialized successfully")


# 모듈 로드 시 초기화 실행
_initialize_module()
