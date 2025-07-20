#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 설정 패키지 초기화 모듈

이 모듈은 전체 시스템의 설정 관리를 담당합니다.
환경별 설정 분리, 전역 설정 인스턴스 관리, 설정 검증 등을 제공합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
버전: 1.0.0
"""
import os
import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

# ====================================================================================
# 1. 패키지 메타데이터
# ====================================================================================

__version__ = "1.0.0"
__author__ = "YOKOGAWA OCR 개발팀"
__email__ = "ocr-dev@yokogawa.com"
__description__ = "YOKOGAWA OCR 데이터 준비 프로젝트 - 설정 관리 패키지"
__license__ = "YOKOGAWA Proprietary"

# ====================================================================================
# 2. 환경 관리 열거형
# ====================================================================================


class Environment(Enum):
    """환경 타입 열거형"""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigurationMode(Enum):
    """설정 모드 열거형"""

    STRICT = "strict"
    LENIENT = "lenient"
    DEBUG = "debug"


# ====================================================================================
# 3. 설정 클래스 및 함수 가져오기 (settings.py)
# ====================================================================================

from .settings import (
    # 설정 클래스들
    ApplicationConfig,
    DatabaseConfig,
    LoggingConfig,
    ProcessingConfig,
    # 설정 관련 함수들
    load_configuration,
    get_database_config,
    get_logging_config,
    validate_configuration,
)

# ====================================================================================
# 4. 상수 정의 가져오기 (constants.py) - 실제 정의된 상수만 사용
# ====================================================================================

from .constants import (
    # 실제 constants.py에 정의된 상수들만 가져오기
    SUPPORTED_FILE_FORMATS,
    DEFAULT_IMAGE_RESOLUTION,
    MAX_FILE_SIZE_MB,
    ANNOTATION_FIELD_TYPES,
    DATA_SPLIT_RATIOS,
    TEXT_MIN_LENGTH,
    TEXT_MAX_LENGTH,
)

# ====================================================================================
# 5. 환경 감지 및 관리 함수
# ====================================================================================


def detect_environment() -> Environment:
    """
    현재 실행 환경 감지

    환경 변수 'YOKOGAWA_OCR_ENV' 또는 'ENVIRONMENT'를 확인하여 환경을 결정합니다.

    Returns:
        Environment: 감지된 환경 타입
    """
    env_var = os.getenv("YOKOGAWA_OCR_ENV", os.getenv("ENVIRONMENT", "development"))

    try:
        return Environment(env_var.lower())
    except ValueError:
        # 알 수 없는 환경인 경우 개발 환경으로 기본 설정
        logging.warning(f"Unknown environment '{env_var}', defaulting to development")
        return Environment.DEVELOPMENT


def is_development_environment() -> bool:
    """개발 환경 여부 확인"""
    return detect_environment() == Environment.DEVELOPMENT


def is_testing_environment() -> bool:
    """테스트 환경 여부 확인"""
    return detect_environment() == Environment.TESTING


def is_staging_environment() -> bool:
    """스테이징 환경 여부 확인"""
    return detect_environment() == Environment.STAGING


def is_production_environment() -> bool:
    """프로덕션 환경 여부 확인"""
    return detect_environment() == Environment.PRODUCTION


def get_config_file_path(environment: Optional[Environment] = None) -> Path:
    """
    환경별 설정 파일 경로 반환

    Args:
        environment: 환경 타입 (None인 경우 자동 감지)

    Returns:
        Path: 설정 파일 경로
    """
    if environment is None:
        environment = detect_environment()

    config_dir = Path(__file__).parent
    config_files = {
        Environment.DEVELOPMENT: "settings_dev.yaml",
        Environment.TESTING: "settings_test.yaml",
        Environment.STAGING: "settings_staging.yaml",
        Environment.PRODUCTION: "settings_prod.yaml",
    }

    config_file = config_files.get(environment, "settings.yaml")
    return config_dir / config_file


def get_environment_variables() -> Dict[str, str]:
    """
    YOKOGAWA OCR 관련 환경 변수 조회

    Returns:
        Dict[str, str]: 환경 변수 딕셔너리
    """
    env_vars = {}
    prefix = "YOKOGAWA_OCR_"

    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_vars[key] = value

    return env_vars


def set_environment_variable(key: str, value: str) -> None:
    """
    환경 변수 설정

    Args:
        key: 환경 변수 키
        value: 환경 변수 값
    """
    full_key = f"YOKOGAWA_OCR_{key.upper()}"
    os.environ[full_key] = value


# ====================================================================================
# 6. 전역 설정 인스턴스 (싱글톤 패턴)
# ====================================================================================


class ConfigManager:
    """
    설정 관리자 클래스 (싱글톤 패턴)

    전역 설정 인스턴스를 관리하고 환경별 설정을 로드합니다.
    """

    _instance: Optional["ConfigManager"] = None
    _application_config: Optional[ApplicationConfig] = None
    _database_config: Optional[DatabaseConfig] = None
    _logging_config: Optional[LoggingConfig] = None
    _processing_config: Optional[ProcessingConfig] = None
    _environment: Optional[Environment] = None
    _is_initialized: bool = False

    def __new__(cls) -> "ConfigManager":
        """싱글톤 인스턴스 생성"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """ConfigManager 초기화"""
        if not self._is_initialized:
            self._environment = detect_environment()
            self._load_configurations()
            self._is_initialized = True

    def _load_configurations(self) -> None:
        """모든 설정 로드"""
        try:
            # 메인 애플리케이션 설정 로드
            self._application_config = load_configuration()

            # 개별 설정 로드
            self._database_config = get_database_config()
            self._logging_config = get_logging_config()
            self._processing_config = self._create_processing_config()

            # 설정 검증
            if not validate_configuration(self._application_config):
                raise ValueError("Configuration validation failed")

            logging.info(
                f"✅ Configuration loaded successfully for {self._environment.value} environment"
            )

        except Exception as e:
            logging.error(f"❌ Failed to load configuration: {str(e)}")
            raise

    def _create_processing_config(self) -> ProcessingConfig:
        """처리 설정 생성 (기본값 사용)"""
        return ProcessingConfig()

    @property
    def application_config(self) -> ApplicationConfig:
        """애플리케이션 설정 반환"""
        if self._application_config is None:
            raise RuntimeError("Application configuration not loaded")
        return self._application_config

    @property
    def database_config(self) -> DatabaseConfig:
        """데이터베이스 설정 반환"""
        if self._database_config is None:
            raise RuntimeError("Database configuration not loaded")
        return self._database_config

    @property
    def logging_config(self) -> LoggingConfig:
        """로깅 설정 반환"""
        if self._logging_config is None:
            raise RuntimeError("Logging configuration not loaded")
        return self._logging_config

    @property
    def processing_config(self) -> ProcessingConfig:
        """처리 설정 반환"""
        if self._processing_config is None:
            raise RuntimeError("Processing configuration not loaded")
        return self._processing_config

    @property
    def environment(self) -> Environment:
        """현재 환경 반환"""
        return self._environment

    def reload_configuration(self) -> None:
        """설정 재로드"""
        self._is_initialized = False
        self._load_configurations()
        self._is_initialized = True
        logging.info("Configuration reloaded successfully")

    def get_config_summary(self) -> Dict[str, Any]:
        """
        설정 요약 정보 반환

        Returns:
            Dict[str, Any]: 설정 요약 정보
        """
        return {
            "environment": self._environment.value,
            "is_initialized": self._is_initialized,
            "supported_file_formats": SUPPORTED_FILE_FORMATS,
            "default_image_resolution": DEFAULT_IMAGE_RESOLUTION,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "annotation_field_types": ANNOTATION_FIELD_TYPES,
            "data_split_ratios": DATA_SPLIT_RATIOS,
            "application_config": {
                "version": getattr(self._application_config, "version", "unknown"),
                "debug_mode": getattr(self._application_config, "debug_mode", False),
            },
            "database_config": {
                "database_type": getattr(
                    self._database_config, "database_type", "unknown"
                ),
                "connection_pool_size": getattr(
                    self._database_config, "connection_pool_size", 0
                ),
            },
            "logging_config": {
                "log_level": getattr(self._logging_config, "log_level", "INFO"),
                "log_format": getattr(self._logging_config, "log_format", "standard"),
            },
            "processing_config": {
                "batch_size": getattr(self._processing_config, "batch_size", 0),
                "thread_count": getattr(self._processing_config, "thread_count", 0),
            },
        }


# ====================================================================================
# 7. 전역 설정 인스턴스 생성 및 접근 함수
# ====================================================================================

# 전역 설정 관리자 인스턴스 생성
_config_manager = ConfigManager()


def get_application_config() -> ApplicationConfig:
    """
    전역 애플리케이션 설정 반환

    Returns:
        ApplicationConfig: 애플리케이션 설정 인스턴스
    """
    return _config_manager.application_config


def get_global_database_config() -> DatabaseConfig:
    """
    전역 데이터베이스 설정 반환

    Returns:
        DatabaseConfig: 데이터베이스 설정 인스턴스
    """
    return _config_manager.database_config


def get_global_logging_config() -> LoggingConfig:
    """
    전역 로깅 설정 반환

    Returns:
        LoggingConfig: 로깅 설정 인스턴스
    """
    return _config_manager.logging_config


def get_global_processing_config() -> ProcessingConfig:
    """
    전역 처리 설정 반환

    Returns:
        ProcessingConfig: 처리 설정 인스턴스
    """
    return _config_manager.processing_config


def get_current_environment() -> Environment:
    """
    현재 환경 반환

    Returns:
        Environment: 현재 환경 타입
    """
    return _config_manager.environment


def reload_global_configuration() -> None:
    """전역 설정 재로드"""
    _config_manager.reload_configuration()


def get_global_config_summary() -> Dict[str, Any]:
    """
    전역 설정 요약 정보 반환

    Returns:
        Dict[str, Any]: 설정 요약 정보
    """
    return _config_manager.get_config_summary()


# ====================================================================================
# 8. 상수 기반 설정 검증 및 유틸리티 함수
# ====================================================================================


def validate_file_format(file_path: str) -> bool:
    """
    파일 형식 검증

    Args:
        file_path: 파일 경로

    Returns:
        bool: 지원되는 파일 형식 여부
    """
    if not file_path:
        return False

    file_extension = Path(file_path).suffix.lower()
    return file_extension in SUPPORTED_FILE_FORMATS


def validate_file_size(file_path: str) -> bool:
    """
    파일 크기 검증

    Args:
        file_path: 파일 경로

    Returns:
        bool: 허용되는 파일 크기 여부
    """
    try:
        if not os.path.exists(file_path):
            return False

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        return file_size_mb <= MAX_FILE_SIZE_MB
    except Exception:
        return False


def get_annotation_field_type(field_name: str) -> Optional[str]:
    """
    어노테이션 필드 타입 조회

    Args:
        field_name: 필드 이름

    Returns:
        Optional[str]: 필드 타입 (없으면 None)
    """
    return ANNOTATION_FIELD_TYPES.get(field_name)


def validate_data_split_ratios() -> bool:
    """
    데이터 분할 비율 검증

    Returns:
        bool: 비율 합계가 1.0인지 여부
    """
    try:
        total_ratio = sum(DATA_SPLIT_RATIOS.values())
        return abs(total_ratio - 1.0) < 0.01  # 소수점 오차 허용
    except Exception:
        return False


def get_default_image_settings() -> Dict[str, Any]:
    """
    기본 이미지 설정 반환

    Returns:
        Dict[str, Any]: 기본 이미지 설정
    """
    return {
        "resolution": DEFAULT_IMAGE_RESOLUTION,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "supported_formats": SUPPORTED_FILE_FORMATS,
    }


def validate_environment_setup() -> bool:
    """
    환경 설정 검증

    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 필수 환경 변수 확인
        required_env_vars = [
            "YOKOGAWA_OCR_DATA_PATH",
            "YOKOGAWA_OCR_LOG_PATH",
            "YOKOGAWA_OCR_TEMP_PATH",
        ]

        missing_vars = []
        for var in required_env_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            # 누락된 환경변수에 대한 기본값 설정
            default_values = {
                "YOKOGAWA_OCR_DATA_PATH": os.path.join(os.getcwd(), "data"),
                "YOKOGAWA_OCR_LOG_PATH": os.path.join(os.getcwd(), "logs"),
                "YOKOGAWA_OCR_TEMP_PATH": os.path.join(os.getcwd(), "temp"),
            }

            for var in missing_vars:
                if var in default_values:
                    os.environ[var] = default_values[var]
                    # 디렉터리 생성
                    os.makedirs(default_values[var], exist_ok=True)
                    logging.info(f"환경변수 {var} 기본값 설정: {default_values[var]}")
            logging.warning(f"Missing environment variables set to default values: {missing_vars}")

            # 재검증
            if any(not os.getenv(var) for var in required_env_vars):
                logging.error(f"Failed to set default values for: {missing_vars}")
                return False

        # 상수 값 검증
        if not SUPPORTED_FILE_FORMATS:
            logging.error("SUPPORTED_FILE_FORMATS is empty")
            return False
        if MAX_FILE_SIZE_MB <= 0:
            logging.error("MAX_FILE_SIZE_MB must be positive")
            return False
        if DEFAULT_IMAGE_RESOLUTION <= 0:
            logging.error("DEFAULT_IMAGE_RESOLUTION must be positive")
            return False
        if not validate_data_split_ratios():
            logging.error("DATA_SPLIT_RATIOS do not sum to 1.0")
            return False

        return True
    except Exception as e:
        logging.error(f"Environment validation failed: {str(e)}")
        return False


def get_system_info() -> Dict[str, Any]:
    """
    시스템 정보 반환

    Returns:
        Dict[str, Any]: 시스템 정보
    """
    return {
        "python_version": sys.version,
        "platform": sys.platform,
        "current_working_directory": os.getcwd(),
        "config_package_version": __version__,
        "environment": get_current_environment().value,
        "environment_variables": get_environment_variables(),
        "constants": {
            "supported_file_formats": SUPPORTED_FILE_FORMATS,
            "default_image_resolution": DEFAULT_IMAGE_RESOLUTION,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "annotation_field_types": ANNOTATION_FIELD_TYPES,
            "data_split_ratios": DATA_SPLIT_RATIOS,
        },
    }


# ====================================================================================
# 9. 패키지 초기화 함수
# ====================================================================================


def initialize_config_package() -> bool:
    """
    설정 패키지 초기화

    Returns:
        bool: 초기화 성공 여부
    """
    try:
        # 환경 설정 검증
        if not validate_environment_setup():
            logging.warning(
                "Environment validation failed, continuing with default settings"
            )

        # 전역 설정 인스턴스 초기화 확인
        if not _config_manager._is_initialized:
            logging.error("Configuration manager initialization failed")
            return False

        # 상수 기반 검증
        if not validate_data_split_ratios():
            logging.error("Data split ratios validation failed")
            return False

        logging.info("✅ Configuration package initialized successfully")
        return True

    except Exception as e:
        logging.error(f"❌ Configuration package initialization failed: {str(e)}")
        return False


# ====================================================================================
# 10. 패키지 레벨 __all__ 정의
# ====================================================================================

__all__ = [
    # 패키지 메타데이터
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__license__",
    # 열거형
    "Environment",
    "ConfigurationMode",
    # 설정 클래스
    "ApplicationConfig",
    "DatabaseConfig",
    "LoggingConfig",
    "ProcessingConfig",
    # 설정 함수
    "load_configuration",
    "get_database_config",
    "get_logging_config",
    "validate_configuration",
    # 실제 constants.py에 정의된 상수들
    "SUPPORTED_FILE_FORMATS",
    "DEFAULT_IMAGE_RESOLUTION",
    "MAX_FILE_SIZE_MB",
    "ANNOTATION_FIELD_TYPES",
    "DATA_SPLIT_RATIOS",
    # 환경 관리 함수
    "detect_environment",
    "is_development_environment",
    "is_testing_environment",
    "is_staging_environment",
    "is_production_environment",
    "get_config_file_path",
    "get_environment_variables",
    "set_environment_variable",
    # 전역 설정 함수
    "get_application_config",
    "get_global_database_config",
    "get_global_logging_config",
    "get_global_processing_config",
    "get_current_environment",
    "reload_global_configuration",
    "get_global_config_summary",
    # 상수 기반 검증 및 유틸리티 함수
    "validate_file_format",
    "validate_file_size",
    "get_annotation_field_type",
    "validate_data_split_ratios",
    "get_default_image_settings",
    "validate_environment_setup",
    "get_system_info",
    "initialize_config_package",
    # 설정 관리자 클래스
    "ConfigManager",
]

# ====================================================================================
# 11. 패키지 로드 시 자동 초기화
# ====================================================================================

# 패키지 로드 시 자동으로 초기화 실행
_initialization_result = initialize_config_package()

if not _initialization_result:
    import warnings

    warnings.warn(
        "Configuration package initialization failed. Some features may not work correctly.",
        RuntimeWarning,
        stacklevel=2,
    )

# 초기화 성공 메시지 (개발 환경에서만 표시)
if is_development_environment():
    print(f"✅ YOKOGAWA OCR Config Package v{__version__} initialized successfully")
    print(f"   - Environment: {get_current_environment().value}")
    print(
        f"   - Configuration Classes: {len([ApplicationConfig, DatabaseConfig, LoggingConfig, ProcessingConfig])}"
    )
    print(
        f"   - Constants Loaded: {len([SUPPORTED_FILE_FORMATS, DEFAULT_IMAGE_RESOLUTION, MAX_FILE_SIZE_MB, ANNOTATION_FIELD_TYPES, DATA_SPLIT_RATIOS])}"
    )
    print(
        f"   - Global Config Manager: {'✅ Ready' if _config_manager._is_initialized else '❌ Failed'}"
    )
