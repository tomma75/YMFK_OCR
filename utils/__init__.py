#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - Utils 패키지 초기화 모듈

이 모듈은 utils 패키지의 초기화를 담당하며,
자주 사용되는 유틸리티 클래스와 함수들을 패키지 레벨에서 노출합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import logging
from typing import Any, Dict, List, Optional, Type, Union

# 버전 정보
__version__ = "1.0.0"
__author__ = "YOKOGAWA OCR 개발팀"
__email__ = "ocr-dev@yokogawa.com"
__description__ = "YOKOGAWA OCR 데이터 준비 프로젝트 - 유틸리티 패키지"

# 패키지 메타데이터
__all__ = [
    # 파일 처리 클래스
    "FileHandler",
    "PDFProcessor",
    "ImageProcessor",
    "JSONProcessor",
    "CompressionHandler",
    # 로깅 클래스
    "CustomLogger",
    "LogFormatter",
    "StructuredLogFormatter",
    "FileRotatingHandler",
    "LoggingManager",
    "LogLevel",
    "LoggerType",
    # 유틸리티 함수
    "create_directory_if_not_exists",
    "copy_file_with_backup",
    "get_file_size_mb",
    "calculate_file_hash",
    "compress_file",
    "extract_compressed_file",
    "get_file_metadata",
    "validate_file_integrity",
    "process_pdf_file",
    "process_image_file",
    "process_json_file",
    # 로깅 함수
    "setup_logger",
    "get_application_logger",
    "initialize_logging",
    "shutdown_logging",
    "create_file_handler",
    "create_console_handler",
    "configure_logging",
    # 패키지 유틸리티
    "get_package_version",
    "validate_utils_dependencies",
    "initialize_utils_package",
    "cleanup_utils_package",
]

# ====================================================================================
# 1. 핵심 모듈 Import
# ====================================================================================

try:
    # 파일 처리 모듈
    from .file_handler import (
        FileHandler,
        PDFProcessor,
        ImageProcessor,
        JSONProcessor,
        CompressionHandler,
        # 유틸리티 함수들
        create_directory_if_not_exists,
        copy_file_with_backup,
        get_file_size_mb,
        calculate_file_hash,
        compress_file,
        extract_compressed_file,
        get_file_metadata,
        validate_file_integrity,
        process_pdf_file,
        process_image_file,
        process_json_file,
    )

    # 로깅 모듈
    from .logger_util import (
        CustomLogger,
        LogFormatter,
        StructuredLogFormatter,
        FileRotatingHandler,
        DatabaseLogHandler,
        LoggingManager,
        LogLevel,
        LoggerType,
        # 로깅 함수들
        setup_logger,
        get_application_logger,
        initialize_logging,
        shutdown_logging,
        create_file_handler,
        create_console_handler,
        configure_logging,
        log_execution_time,
        log_method_calls,
        format_log_message,
    )

    # 선택적 모듈 import (존재하는 경우만)
    try:
        from .image_processor import (
            ImageConverter,
            ImageEnhancer,
            ImageValidator,
            convert_pdf_to_images,
            resize_image,
            enhance_image_quality,
            detect_image_orientation,
            correct_image_skew,
            validate_image_format,
        )

        # 이미지 처리 관련 요소를 __all__에 추가
        __all__.extend(
            [
                "ImageConverter",
                "ImageEnhancer",
                "ImageValidator",
                "convert_pdf_to_images",
                "resize_image",
                "enhance_image_quality",
                "detect_image_orientation",
                "correct_image_skew",
                "validate_image_format",
            ]
        )

    except ImportError:
        # 이미지 처리 모듈이 없는 경우 로깅
        logging.warning(
            "Image processor module not found - image processing features will be limited"
        )

except ImportError as e:
    logging.error(f"Failed to import utils modules: {e}")
    raise ImportError(f"Utils package initialization failed: {e}")

# ====================================================================================
# 2. 패키지 레벨 유틸리티 함수
# ====================================================================================


def get_package_version() -> str:
    """
    패키지 버전 정보 반환

    Returns:
        str: 패키지 버전 문자열
    """
    return __version__


def validate_utils_dependencies() -> Dict[str, bool]:
    """
    Utils 패키지 의존성 검증

    Returns:
        Dict[str, bool]: 의존성 검증 결과
    """
    dependencies_status = {}

    # 필수 의존성 확인
    required_modules = [
        "os",
        "json",
        "logging",
        "pathlib",
        "datetime",
        "hashlib",
        "shutil",
        "tempfile",
        "uuid",
        "typing",
    ]

    for module_name in required_modules:
        try:
            __import__(module_name)
            dependencies_status[module_name] = True
        except ImportError:
            dependencies_status[module_name] = False

    # 선택적 의존성 확인
    optional_modules = [
        "fitz",  # PyMuPDF
        "PIL",  # Pillow
        "numpy",  # NumPy
        "cv2",  # OpenCV
    ]

    for module_name in optional_modules:
        try:
            __import__(module_name)
            dependencies_status[f"{module_name}_optional"] = True
        except ImportError:
            dependencies_status[f"{module_name}_optional"] = False

    return dependencies_status


def initialize_utils_package(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Utils 패키지 초기화

    Args:
        config: 초기화 설정 딕셔너리

    Returns:
        bool: 초기화 성공 여부
    """
    try:
        # 기본 설정
        default_config = {
            "enable_logging": True,
            "log_level": "INFO",
            "validate_dependencies": True,
            "temp_directory": None,
        }

        # 설정 병합
        if config:
            default_config.update(config)

        # 의존성 검증
        if default_config["validate_dependencies"]:
            deps_status = validate_utils_dependencies()
            failed_deps = [
                dep
                for dep, status in deps_status.items()
                if not status and not dep.endswith("_optional")
            ]

            if failed_deps:
                logging.error(f"Missing required dependencies: {failed_deps}")
                return False

        # 로깅 초기화
        if default_config["enable_logging"]:
            logging.basicConfig(
                level=getattr(logging, default_config["log_level"]),
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )

        # 임시 디렉터리 설정
        if default_config["temp_directory"]:
            import tempfile

            tempfile.tempdir = default_config["temp_directory"]

        logging.info(f"Utils package initialized successfully - Version: {__version__}")
        return True

    except Exception as e:
        logging.error(f"Utils package initialization failed: {e}")
        return False


def cleanup_utils_package() -> None:
    """
    Utils 패키지 정리
    """
    try:
        # 로깅 시스템 종료
        if "shutdown_logging" in globals():
            shutdown_logging()

        # 임시 파일 정리
        if "FileHandler" in globals():
            handler = FileHandler()
            handler.cleanup_temp_files()

        logging.info("Utils package cleanup completed")

    except Exception as e:
        logging.error(f"Utils package cleanup failed: {e}")


# ====================================================================================
# 3. 패키지 레벨 편의 함수
# ====================================================================================


def create_default_file_handler(config: Optional[Dict[str, Any]] = None) -> FileHandler:
    """
    기본 설정으로 FileHandler 인스턴스 생성

    Args:
        config: 파일 핸들러 설정

    Returns:
        FileHandler: 파일 핸들러 인스턴스
    """
    try:
        return FileHandler(config)
    except Exception as e:
        logging.error(f"Failed to create FileHandler: {e}")
        raise


def create_default_logger(
    name: str,
    level: str = "INFO",
    enable_file_logging: bool = True,
    log_file_path: Optional[str] = None,
) -> CustomLogger:
    """
    기본 설정으로 CustomLogger 인스턴스 생성

    Args:
        name: 로거 이름
        level: 로그 레벨
        enable_file_logging: 파일 로깅 활성화 여부
        log_file_path: 로그 파일 경로

    Returns:
        CustomLogger: 커스텀 로거 인스턴스
    """
    try:
        # 기본 로깅 설정 생성
        from config.settings import LoggingConfig

        logging_config = LoggingConfig()
        logging_config.log_level = level
        logging_config.enable_file_logging = enable_file_logging

        if log_file_path:
            logging_config.log_file_path = log_file_path

        return setup_logger(name, logging_config)

    except Exception as e:
        logging.error(f"Failed to create CustomLogger: {e}")
        raise


def process_file_by_type(
    file_path: str,
    file_type: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    파일 유형에 따라 적절한 처리 수행

    Args:
        file_path: 처리할 파일 경로
        file_type: 파일 유형 (자동 감지 시 None)
        config: 처리 설정

    Returns:
        Dict[str, Any]: 처리 결과
    """
    try:
        from pathlib import Path

        if not file_type:
            # 파일 확장자로 유형 판단
            file_extension = Path(file_path).suffix.lower()

            if file_extension in [".pdf"]:
                file_type = "pdf"
            elif file_extension in [".jpg", ".jpeg", ".png", ".tiff", ".bmp"]:
                file_type = "image"
            elif file_extension in [".json"]:
                file_type = "json"
            else:
                file_type = "unknown"

        # 설정 객체 생성
        if config is None:
            from config.settings import ApplicationConfig

            app_config = ApplicationConfig()
        else:
            app_config = config

        # 파일 유형별 처리
        if file_type == "pdf":
            return process_pdf_file(file_path, app_config)
        elif file_type == "image":
            return process_image_file(file_path, app_config)
        elif file_type == "json":
            return process_json_file(file_path, app_config)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        logging.error(f"Failed to process file {file_path}: {e}")
        raise


# ====================================================================================
# 4. 패키지 초기화 실행
# ====================================================================================

# 패키지 로드 시 자동 초기화
try:
    if not initialize_utils_package():
        logging.warning("Utils package initialization completed with warnings")
except Exception as e:
    logging.error(f"Utils package auto-initialization failed: {e}")

# ====================================================================================
# 5. 패키지 정보 출력 (개발 모드에서만)
# ====================================================================================


def _print_package_info() -> None:
    """패키지 정보 출력 (개발용)"""
    print(
        f"""
    ==========================================
    YOKOGAWA OCR Utils Package
    ==========================================
    Version: {__version__}
    Author: {__author__}
    Description: {__description__}
    
    Available Classes:
    - FileHandler, PDFProcessor, ImageProcessor, JSONProcessor
    - CustomLogger, LogFormatter, LoggingManager
    
    Available Functions:
    - File processing utilities
    - Logging utilities
    - Package management functions
    ==========================================
    """
    )


# 개발 모드에서만 정보 출력
if __name__ == "__main__":
    _print_package_info()
