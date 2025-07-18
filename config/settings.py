#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 설정 관리 모듈

이 모듈은 전체 애플리케이션의 설정을 관리하며, 의존성 주입 패턴을 통해
각 서비스에 필요한 설정 정보를 제공합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-01-18
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from pathlib import Path
from datetime import timedelta

from core.exceptions import ConfigurationError


@dataclass
class DatabaseConfig:
    """데이터베이스 설정 클래스"""
    
    # 연결 정보
    host: str = "localhost"
    port: int = 5432
    database_name: str = "yokogawa_ocr"
    username: str = "postgres"
    password: str = ""
    
    # 연결 풀 설정
    max_connections: int = 20
    min_connections: int = 5
    connection_timeout: int = 30
    
    # 성능 설정
    statement_timeout: int = 300
    idle_timeout: int = 600
    
    def __post_init__(self) -> None:
        """설정 후 초기화 검증"""
        self._validate_database_config()
    
    def _validate_database_config(self) -> None:
        """데이터베이스 설정 검증"""
        if not self.host:
            raise ConfigurationError("Database host cannot be empty")
        
        if not (1 <= self.port <= 65535):
            raise ConfigurationError(f"Invalid database port: {self.port}")
        
        if not self.database_name:
            raise ConfigurationError("Database name cannot be empty")
    
    def get_connection_string(self) -> str:
        """데이터베이스 연결 문자열 생성"""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database_name}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "host": self.host,
            "port": self.port,
            "database_name": self.database_name,
            "username": self.username,
            "max_connections": self.max_connections,
            "min_connections": self.min_connections,
            "connection_timeout": self.connection_timeout,
            "statement_timeout": self.statement_timeout,
            "idle_timeout": self.idle_timeout
        }


@dataclass
class LoggingConfig:
    """로깅 설정 클래스"""
    
    # 로그 레벨 설정
    log_level: str = "INFO"
    console_log_level: str = "INFO"
    file_log_level: str = "DEBUG"
    
    # 로그 파일 설정
    log_file_path: str = "logs/yokogawa_ocr.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    
    # 로그 포맷 설정
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 로그 필터 설정
    enable_console_logging: bool = True
    enable_file_logging: bool = True
    enable_structured_logging: bool = True
    
    def __post_init__(self) -> None:
        """설정 후 초기화 검증"""
        self._validate_logging_config()
        self._ensure_log_directory()
    
    def _validate_logging_config(self) -> None:
        """로깅 설정 검증"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        
        if self.log_level not in valid_levels:
            raise ConfigurationError(f"Invalid log level: {self.log_level}")
        
        if self.console_log_level not in valid_levels:
            raise ConfigurationError(f"Invalid console log level: {self.console_log_level}")
        
        if self.file_log_level not in valid_levels:
            raise ConfigurationError(f"Invalid file log level: {self.file_log_level}")
    
    def _ensure_log_directory(self) -> None:
        """로그 디렉터리 생성"""
        log_dir = Path(self.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    def get_log_level_numeric(self) -> int:
        """로그 레벨을 숫자로 변환"""
        return getattr(logging, self.log_level)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "log_level": self.log_level,
            "console_log_level": self.console_log_level,
            "file_log_level": self.file_log_level,
            "log_file_path": self.log_file_path,
            "max_file_size_mb": self.max_file_size_mb,
            "backup_count": self.backup_count,
            "log_format": self.log_format,
            "date_format": self.date_format,
            "enable_console_logging": self.enable_console_logging,
            "enable_file_logging": self.enable_file_logging,
            "enable_structured_logging": self.enable_structured_logging
        }


@dataclass
class ProcessingConfig:
    """데이터 처리 설정 클래스"""
    
    # 파일 처리 설정
    supported_file_formats: List[str] = field(default_factory=lambda: [".pdf", ".png", ".jpg", ".jpeg"])
    max_file_size_mb: int = 500
    batch_size: int = 10
    
    # 이미지 처리 설정
    default_image_resolution: int = 300
    image_quality: int = 95
    enable_image_preprocessing: bool = True
    
    # PDF 처리 설정
    pdf_extraction_mode: str = "text_and_images"
    pdf_password: Optional[str] = None
    extract_embedded_images: bool = True
    
    # 병렬 처리 설정
    max_workers: int = 4
    enable_multiprocessing: bool = True
    processing_timeout_seconds: int = 300
    
    # 데이터 분할 설정
    train_split_ratio: float = 0.7
    validation_split_ratio: float = 0.2
    test_split_ratio: float = 0.1
    
    # 데이터 증강 설정
    enable_data_augmentation: bool = True
    augmentation_factor: int = 3
    max_rotation_degrees: int = 5
    max_scale_factor: float = 0.1
    
    def __post_init__(self) -> None:
        """설정 후 초기화 검증"""
        self._validate_processing_config()
    
    def _validate_processing_config(self) -> None:
        """처리 설정 검증"""
        # 파일 크기 검증
        if self.max_file_size_mb <= 0:
            raise ConfigurationError("Max file size must be positive")
        
        # 배치 크기 검증
        if self.batch_size <= 0:
            raise ConfigurationError("Batch size must be positive")
        
        # 해상도 검증
        if self.default_image_resolution <= 0:
            raise ConfigurationError("Image resolution must be positive")
        
        # 분할 비율 검증
        total_ratio = self.train_split_ratio + self.validation_split_ratio + self.test_split_ratio
        if abs(total_ratio - 1.0) > 0.001:
            raise ConfigurationError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # 워커 수 검증
        if self.max_workers <= 0:
            raise ConfigurationError("Max workers must be positive")
    
    def get_split_ratios(self) -> Dict[str, float]:
        """데이터 분할 비율 반환"""
        return {
            "train": self.train_split_ratio,
            "validation": self.validation_split_ratio,
            "test": self.test_split_ratio
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "supported_file_formats": self.supported_file_formats,
            "max_file_size_mb": self.max_file_size_mb,
            "batch_size": self.batch_size,
            "default_image_resolution": self.default_image_resolution,
            "image_quality": self.image_quality,
            "enable_image_preprocessing": self.enable_image_preprocessing,
            "pdf_extraction_mode": self.pdf_extraction_mode,
            "extract_embedded_images": self.extract_embedded_images,
            "max_workers": self.max_workers,
            "enable_multiprocessing": self.enable_multiprocessing,
            "processing_timeout_seconds": self.processing_timeout_seconds,
            "train_split_ratio": self.train_split_ratio,
            "validation_split_ratio": self.validation_split_ratio,
            "test_split_ratio": self.test_split_ratio,
            "enable_data_augmentation": self.enable_data_augmentation,
            "augmentation_factor": self.augmentation_factor,
            "max_rotation_degrees": self.max_rotation_degrees,
            "max_scale_factor": self.max_scale_factor
        }


@dataclass
class ApplicationConfig:
    """메인 애플리케이션 설정 클래스"""
    
    # 애플리케이션 정보
    app_name: str = "YOKOGAWA OCR Data Preparation"
    app_version: str = "1.0.0"
    app_description: str = "OCR 데이터 준비 및 전처리 시스템"
    
    # 환경 설정
    environment: str = "development"
    debug_mode: bool = False
    
    # 디렉터리 경로 설정
    base_directory: str = "."
    data_directory: str = "data"
    raw_data_directory: str = "data/raw"
    processed_data_directory: str = "data/processed"
    annotations_directory: str = "data/annotations"
    augmented_data_directory: str = "data/augmented"
    templates_directory: str = "templates"
    
    # 하위 설정 객체들
    database_config: DatabaseConfig = field(default_factory=DatabaseConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    processing_config: ProcessingConfig = field(default_factory=ProcessingConfig)
    
    def __post_init__(self) -> None:
        """설정 후 초기화"""
        self._validate_application_config()
        self._ensure_directories()
    
    def _validate_application_config(self) -> None:
        """애플리케이션 설정 검증"""
        if not self.app_name:
            raise ConfigurationError("Application name cannot be empty")
        
        if not self.app_version:
            raise ConfigurationError("Application version cannot be empty")
        
        valid_environments = ["development", "testing", "production"]
        if self.environment not in valid_environments:
            raise ConfigurationError(f"Invalid environment: {self.environment}")
    
    def _ensure_directories(self) -> None:
        """필요한 디렉터리 생성"""
        directories = [
            self.data_directory,
            self.raw_data_directory,
            self.processed_data_directory,
            self.annotations_directory,
            self.augmented_data_directory,
            self.templates_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def is_production(self) -> bool:
        """프로덕션 환경 여부 확인"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """개발 환경 여부 확인"""
        return self.environment == "development"
    
    def get_full_path(self, relative_path: str) -> str:
        """상대 경로를 절대 경로로 변환"""
        return os.path.join(self.base_directory, relative_path)
    
    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            "app_name": self.app_name,
            "app_version": self.app_version,
            "app_description": self.app_description,
            "environment": self.environment,
            "debug_mode": self.debug_mode,
            "base_directory": self.base_directory,
            "data_directory": self.data_directory,
            "raw_data_directory": self.raw_data_directory,
            "processed_data_directory": self.processed_data_directory,
            "annotations_directory": self.annotations_directory,
            "augmented_data_directory": self.augmented_data_directory,
            "templates_directory": self.templates_directory,
            "database_config": self.database_config.to_dict(),
            "logging_config": self.logging_config.to_dict(),
            "processing_config": self.processing_config.to_dict()
        }


# 전역 설정 인스턴스
_application_config: Optional[ApplicationConfig] = None


def load_configuration(config_file_path: Optional[str] = None) -> ApplicationConfig:
    """
    설정 파일을 로드하고 ApplicationConfig 인스턴스를 생성합니다.
    
    Args:
        config_file_path: 설정 파일 경로 (None인 경우 환경변수 또는 기본값 사용)
    
    Returns:
        ApplicationConfig: 로드된 설정 인스턴스
    
    Raises:
        ConfigurationError: 설정 로드 또는 검증 실패 시
    """
    global _application_config
    
    if _application_config is not None:
        return _application_config
    
    try:
        # 환경변수에서 설정 파일 경로 확인
        if config_file_path is None:
            config_file_path = os.getenv("YOKOGAWA_CONFIG_FILE", "config/application.json")
        
        # 설정 파일 존재 확인
        if os.path.exists(config_file_path):
            with open(config_file_path, 'r', encoding='utf-8') as file:
                config_data = json.load(file)
            _application_config = _create_config_from_dict(config_data)
        else:
            # 기본 설정 사용
            _application_config = ApplicationConfig()
        
        # 환경변수로 설정 오버라이드
        _override_config_from_environment(_application_config)
        
        # 설정 검증
        if not validate_configuration(_application_config):
            raise ConfigurationError("Configuration validation failed")
        
        return _application_config
        
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {str(e)}")


def _create_config_from_dict(config_data: Dict[str, Any]) -> ApplicationConfig:
    """딕셔너리에서 설정 객체 생성"""
    
    # 데이터베이스 설정
    db_config = DatabaseConfig()
    if "database_config" in config_data:
        db_data = config_data["database_config"]
        db_config = DatabaseConfig(**db_data)
    
    # 로깅 설정
    logging_config = LoggingConfig()
    if "logging_config" in config_data:
        logging_data = config_data["logging_config"]
        logging_config = LoggingConfig(**logging_data)
    
    # 처리 설정
    processing_config = ProcessingConfig()
    if "processing_config" in config_data:
        processing_data = config_data["processing_config"]
        processing_config = ProcessingConfig(**processing_data)
    
    # 메인 애플리케이션 설정
    app_config_data = {k: v for k, v in config_data.items() 
                       if k not in ["database_config", "logging_config", "processing_config"]}
    
    return ApplicationConfig(
        database_config=db_config,
        logging_config=logging_config,
        processing_config=processing_config,
        **app_config_data
    )


def _override_config_from_environment(config: ApplicationConfig) -> None:
    """환경변수로 설정 오버라이드"""
    
    # 애플리케이션 설정 오버라이드
    if os.getenv("YOKOGAWA_ENVIRONMENT"):
        config.environment = os.getenv("YOKOGAWA_ENVIRONMENT")
    
    if os.getenv("YOKOGAWA_DEBUG_MODE"):
        config.debug_mode = os.getenv("YOKOGAWA_DEBUG_MODE").lower() == "true"
    
    # 데이터베이스 설정 오버라이드
    if os.getenv("YOKOGAWA_DB_HOST"):
        config.database_config.host = os.getenv("YOKOGAWA_DB_HOST")
    
    if os.getenv("YOKOGAWA_DB_PORT"):
        config.database_config.port = int(os.getenv("YOKOGAWA_DB_PORT"))
    
    if os.getenv("YOKOGAWA_DB_NAME"):
        config.database_config.database_name = os.getenv("YOKOGAWA_DB_NAME")
    
    if os.getenv("YOKOGAWA_DB_USERNAME"):
        config.database_config.username = os.getenv("YOKOGAWA_DB_USERNAME")
    
    if os.getenv("YOKOGAWA_DB_PASSWORD"):
        config.database_config.password = os.getenv("YOKOGAWA_DB_PASSWORD")
    
    # 로깅 설정 오버라이드
    if os.getenv("YOKOGAWA_LOG_LEVEL"):
        config.logging_config.log_level = os.getenv("YOKOGAWA_LOG_LEVEL")
    
    if os.getenv("YOKOGAWA_LOG_FILE"):
        config.logging_config.log_file_path = os.getenv("YOKOGAWA_LOG_FILE")


def get_database_config() -> DatabaseConfig:
    """데이터베이스 설정 반환"""
    config = load_configuration()
    return config.database_config


def get_logging_config() -> LoggingConfig:
    """로깅 설정 반환"""
    config = load_configuration()
    return config.logging_config


def get_processing_config() -> ProcessingConfig:
    """처리 설정 반환"""
    config = load_configuration()
    return config.processing_config


def validate_configuration(config: ApplicationConfig) -> bool:
    """
    설정 검증 수행
    
    Args:
        config: 검증할 설정 객체
    
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 각 설정 객체의 내장 검증 실행
        # (이미 __post_init__에서 수행되지만 명시적으로 호출)
        
        # 애플리케이션 설정 검증
        if not config.app_name or not config.app_version:
            return False
        
        # 디렉터리 접근 가능성 확인
        base_path = Path(config.base_directory)
        if not base_path.exists():
            base_path.mkdir(parents=True, exist_ok=True)
        
        # 데이터베이스 설정 검증
        if not config.database_config.host or not config.database_config.database_name:
            return False
        
        # 로깅 설정 검증
        if not config.logging_config.log_level:
            return False
        
        # 처리 설정 검증
        if config.processing_config.max_file_size_mb <= 0:
            return False
        
        return True
        
    except Exception:
        return False


def reload_configuration(config_file_path: Optional[str] = None) -> ApplicationConfig:
    """
    설정을 다시 로드합니다.
    
    Args:
        config_file_path: 설정 파일 경로
    
    Returns:
        ApplicationConfig: 다시 로드된 설정 인스턴스
    """
    global _application_config
    _application_config = None
    return load_configuration(config_file_path)


def get_current_config() -> Optional[ApplicationConfig]:
    """현재 로드된 설정 반환"""
    return _application_config


def save_configuration(config: ApplicationConfig, config_file_path: str) -> None:
    """
    설정을 파일에 저장합니다.
    
    Args:
        config: 저장할 설정 객체
        config_file_path: 저장할 파일 경로
    
    Raises:
        ConfigurationError: 저장 실패 시
    """
    try:
        # 디렉터리 생성
        config_dir = Path(config_file_path).parent
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # 설정 딕셔너리 변환 및 저장
        config_dict = config.to_dict()
        
        with open(config_file_path, 'w', encoding='utf-8') as file:
            json.dump(config_dict, file, indent=4, ensure_ascii=False)
            
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration: {str(e)}")


# 의존성 주입을 위한 설정 팩토리 함수들
def create_application_config(**kwargs) -> ApplicationConfig:
    """ApplicationConfig 인스턴스 생성 팩토리"""
    return ApplicationConfig(**kwargs)


def create_database_config(**kwargs) -> DatabaseConfig:
    """DatabaseConfig 인스턴스 생성 팩토리"""
    return DatabaseConfig(**kwargs)


def create_logging_config(**kwargs) -> LoggingConfig:
    """LoggingConfig 인스턴스 생성 팩토리"""
    return LoggingConfig(**kwargs)


def create_processing_config(**kwargs) -> ProcessingConfig:
    """ProcessingConfig 인스턴스 생성 팩토리"""
    return ProcessingConfig(**kwargs)


if __name__ == "__main__":
    # 설정 테스트 실행
    try:
        config = load_configuration()
        print(f"Configuration loaded successfully: {config.app_name} v{config.app_version}")
        print(f"Environment: {config.environment}")
        print(f"Debug mode: {config.debug_mode}")
        
        # 설정 검증
        if validate_configuration(config):
            print("✅ Configuration validation passed")
        else:
            print("❌ Configuration validation failed")
            
    except ConfigurationError as e:
        print(f"❌ Configuration error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
