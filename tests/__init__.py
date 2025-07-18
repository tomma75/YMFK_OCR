#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 테스트 패키지 초기화 모듈

이 모듈은 테스트 패키지의 초기화를 담당하며,
모든 테스트 모듈에서 공통으로 사용되는 테스트 유틸리티 함수들을 제공합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import uuid

# 프로젝트 루트 디렉터리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 대상 모듈 import
from config.settings import ApplicationConfig, load_configuration
from core.exceptions import *
from services.data_collection_service import create_data_collection_service
from services.labeling_service import create_labeling_service
from services.augmentation_service import create_augmentation_service
from services.validation_service import create_validation_service
from models.document_model import DocumentModel, DocumentMetadata, PageInfo
from models.annotation_model import AnnotationModel, BoundingBox, FieldAnnotation
from utils.logger_util import setup_logger, get_application_logger
from utils.file_handler import FileHandler
from utils.image_processor import ImageProcessor

# ====================================================================================
# 1. 테스트 패키지 메타데이터
# ====================================================================================

__version__ = "1.0.0"
__author__ = "YOKOGAWA OCR 개발팀"
__email__ = "ocr-dev@yokogawa.com"
__description__ = "YOKOGAWA OCR 데이터 준비 프로젝트 - 테스트 패키지"

# ====================================================================================
# 2. 테스트 설정 및 상수
# ====================================================================================

# 테스트 환경 설정
TEST_ENVIRONMENT = "testing"
TEST_LOG_LEVEL = "DEBUG"
TEST_TIMEOUT_SECONDS = 30
TEST_BATCH_SIZE = 5
TEST_MAX_WORKERS = 2

# 테스트 데이터 설정
TEST_DATA_DIR = "test_data"
TEST_PDF_FILE = "test_document.pdf"
TEST_IMAGE_FILE = "test_image.jpg"
TEST_JSON_FILE = "test_data.json"

# 테스트 결과 설정
TEST_REPORT_DIR = "test_reports"
TEST_COVERAGE_MIN = 80.0

# ====================================================================================
# 3. 테스트 헬퍼 함수들
# ====================================================================================


def create_test_config() -> ApplicationConfig:
    """
    테스트용 애플리케이션 설정 생성

    Returns:
        ApplicationConfig: 테스트용 설정 객체
    """
    config = ApplicationConfig()
    config.environment = TEST_ENVIRONMENT
    config.debug_mode = True
    config.data_directory = tempfile.mkdtemp(prefix="yokogawa_test_")
    config.raw_data_directory = os.path.join(config.data_directory, "raw")
    config.processed_data_directory = os.path.join(config.data_directory, "processed")
    config.annotations_directory = os.path.join(config.data_directory, "annotations")
    config.augmented_data_directory = os.path.join(config.data_directory, "augmented")
    config.processing_config.batch_size = TEST_BATCH_SIZE
    config.processing_config.max_workers = TEST_MAX_WORKERS
    config.logging_config.log_level = TEST_LOG_LEVEL
    return config


def create_test_directories(config: ApplicationConfig) -> None:
    """
    테스트용 디렉터리 생성

    Args:
        config: 애플리케이션 설정 객체
    """
    directories = [
        config.data_directory,
        config.raw_data_directory,
        config.processed_data_directory,
        config.annotations_directory,
        config.augmented_data_directory,
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def cleanup_test_directories(config: ApplicationConfig) -> None:
    """
    테스트용 디렉터리 정리

    Args:
        config: 애플리케이션 설정 객체
    """
    if os.path.exists(config.data_directory):
        shutil.rmtree(config.data_directory)


def create_test_logger(name: str = "test_logger") -> logging.Logger:
    """
    테스트용 로거 생성

    Args:
        name: 로거 이름

    Returns:
        logging.Logger: 테스트용 로거 객체
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # 콘솔 핸들러 추가
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


# ====================================================================================
# 4. 테스트 데이터 생성 함수들
# ====================================================================================


def create_test_pdf_file(file_path: str) -> None:
    """
    테스트용 PDF 파일 생성

    Args:
        file_path: 생성할 PDF 파일 경로
    """
    # 간단한 PDF 헤더 생성
    pdf_content = b"""%PDF-1.4
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
2 0 obj
<< /Type /Pages /Kids [3 0 R] /Count 1 >>
endobj
3 0 obj
<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>
endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
trailer
<< /Size 4 /Root 1 0 R >>
startxref
190
%%EOF"""

    with open(file_path, "wb") as f:
        f.write(pdf_content)


def create_test_image_file(file_path: str, width: int = 100, height: int = 100) -> None:
    """
    테스트용 이미지 파일 생성

    Args:
        file_path: 생성할 이미지 파일 경로
        width: 이미지 너비
        height: 이미지 높이
    """
    try:
        from PIL import Image

        # 간단한 테스트 이미지 생성
        image = Image.new("RGB", (width, height), color="white")
        image.save(file_path)
    except ImportError:
        # PIL이 없는 경우 더미 이미지 파일 생성
        dummy_image_data = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d\x00\x00\x00d\x08\x06\x00\x00\x00p\xe2\x95!..."
        with open(file_path, "wb") as f:
            f.write(dummy_image_data)


def create_test_json_file(
    file_path: str, data: Optional[Dict[str, Any]] = None
) -> None:
    """
    테스트용 JSON 파일 생성

    Args:
        file_path: 생성할 JSON 파일 경로
        data: JSON 데이터 (None인 경우 기본 데이터 사용)
    """
    if data is None:
        data = {
            "test_id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "test_type": "unit_test",
            "test_data": {"documents": [], "annotations": []},
        }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def create_test_document_model(file_path: Optional[str] = None) -> DocumentModel:
    """
    테스트용 DocumentModel 생성

    Args:
        file_path: 문서 파일 경로

    Returns:
        DocumentModel: 테스트용 문서 모델
    """
    if file_path is None:
        file_path = create_temp_test_file("test_doc.pdf")
        create_test_pdf_file(file_path)

    return DocumentModel.from_file_path(file_path)


def create_test_annotation_model(document_id: str = "test_doc") -> AnnotationModel:
    """
    테스트용 AnnotationModel 생성

    Args:
        document_id: 문서 ID

    Returns:
        AnnotationModel: 테스트용 어노테이션 모델
    """
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    annotation = AnnotationModel(
        document_id=document_id, page_number=1, annotation_type="text"
    )
    return annotation


def create_test_bounding_box(
    x: int = 10, y: int = 20, width: int = 100, height: int = 50
) -> BoundingBox:
    """
    테스트용 BoundingBox 생성

    Args:
        x: X 좌표
        y: Y 좌표
        width: 너비
        height: 높이

    Returns:
        BoundingBox: 테스트용 바운딩 박스
    """
    return BoundingBox(x=x, y=y, width=width, height=height)


def create_temp_test_file(filename: str) -> str:
    """
    임시 테스트 파일 경로 생성

    Args:
        filename: 파일명

    Returns:
        str: 임시 파일 경로
    """
    temp_dir = tempfile.mkdtemp(prefix="yokogawa_test_")
    return os.path.join(temp_dir, filename)


# ====================================================================================
# 5. 테스트 데이터 검증 함수들
# ====================================================================================


def validate_test_environment() -> bool:
    """
    테스트 환경 검증

    Returns:
        bool: 검증 결과
    """
    try:
        # 필수 모듈 import 확인
        import numpy as np
        import PIL

        # 임시 디렉터리 생성 확인
        temp_dir = tempfile.mkdtemp()
        os.rmdir(temp_dir)

        # 로깅 시스템 확인
        test_logger = create_test_logger()
        test_logger.info("Test environment validation")

        return True
    except Exception as e:
        print(f"Test environment validation failed: {e}")
        return False


def validate_test_data_integrity(data: Any) -> bool:
    """
    테스트 데이터 무결성 검증

    Args:
        data: 검증할 데이터

    Returns:
        bool: 검증 결과
    """
    try:
        if isinstance(data, dict):
            return all(isinstance(key, str) for key in data.keys())
        elif isinstance(data, list):
            return len(data) >= 0
        elif isinstance(data, str):
            return len(data) >= 0
        else:
            return True
    except Exception:
        return False


def compare_test_results(expected: Any, actual: Any, tolerance: float = 0.001) -> bool:
    """
    테스트 결과 비교

    Args:
        expected: 예상 결과
        actual: 실제 결과
        tolerance: 허용 오차 (숫자 비교 시)

    Returns:
        bool: 비교 결과
    """
    try:
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return abs(expected - actual) <= tolerance
        elif isinstance(expected, str) and isinstance(actual, str):
            return expected == actual
        elif isinstance(expected, list) and isinstance(actual, list):
            return len(expected) == len(actual) and all(
                compare_test_results(e, a, tolerance) for e, a in zip(expected, actual)
            )
        elif isinstance(expected, dict) and isinstance(actual, dict):
            return expected.keys() == actual.keys() and all(
                compare_test_results(expected[k], actual[k], tolerance)
                for k in expected.keys()
            )
        else:
            return expected == actual
    except Exception:
        return False


# ====================================================================================
# 6. 테스트 실행 유틸리티 함수들
# ====================================================================================


def run_test_suite(test_class: type, verbosity: int = 2) -> unittest.TestResult:
    """
    테스트 스위트 실행

    Args:
        test_class: 테스트 클래스
        verbosity: 출력 상세도

    Returns:
        unittest.TestResult: 테스트 결과
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(test_class)
    runner = unittest.TextTestRunner(verbosity=verbosity)
    return runner.run(suite)


def measure_test_execution_time(test_func: Callable) -> tuple:
    """
    테스트 실행 시간 측정

    Args:
        test_func: 테스트 함수

    Returns:
        tuple: (실행 시간, 테스트 결과)
    """
    import time

    start_time = time.time()
    try:
        result = test_func()
        success = True
    except Exception as e:
        result = e
        success = False
    end_time = time.time()

    execution_time = end_time - start_time
    return execution_time, result, success


def create_test_report(test_results: Dict[str, Any], output_path: str) -> None:
    """
    테스트 리포트 생성

    Args:
        test_results: 테스트 결과 데이터
        output_path: 리포트 출력 경로
    """
    report_data = {
        "test_report": {
            "timestamp": datetime.now().isoformat(),
            "environment": TEST_ENVIRONMENT,
            "results": test_results,
            "summary": {
                "total_tests": test_results.get("total_tests", 0),
                "passed_tests": test_results.get("passed_tests", 0),
                "failed_tests": test_results.get("failed_tests", 0),
                "coverage": test_results.get("coverage", 0.0),
            },
        }
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)


# ====================================================================================
# 7. 목 객체 생성 함수들
# ====================================================================================


def create_mock_config() -> Mock:
    """
    Mock ApplicationConfig 생성

    Returns:
        Mock: Mock 설정 객체
    """
    mock_config = Mock(spec=ApplicationConfig)
    mock_config.environment = TEST_ENVIRONMENT
    mock_config.debug_mode = True
    mock_config.data_directory = tempfile.mkdtemp()
    mock_config.processing_config.batch_size = TEST_BATCH_SIZE
    mock_config.processing_config.max_workers = TEST_MAX_WORKERS
    return mock_config


def create_mock_logger() -> Mock:
    """
    Mock Logger 생성

    Returns:
        Mock: Mock 로거 객체
    """
    mock_logger = Mock(spec=logging.Logger)
    mock_logger.debug = Mock()
    mock_logger.info = Mock()
    mock_logger.warning = Mock()
    mock_logger.error = Mock()
    mock_logger.critical = Mock()
    return mock_logger


def create_mock_file_handler() -> Mock:
    """
    Mock FileHandler 생성

    Returns:
        Mock: Mock 파일 핸들러 객체
    """
    mock_handler = Mock(spec=FileHandler)
    mock_handler.create_directory_if_not_exists = Mock(return_value=True)
    mock_handler.copy_file_with_backup = Mock(return_value=True)
    mock_handler.get_file_size_mb = Mock(return_value=10.5)
    mock_handler.calculate_file_hash = Mock(return_value="abc123")
    return mock_handler


def create_mock_service(service_type: str) -> Mock:
    """
    Mock 서비스 생성

    Args:
        service_type: 서비스 타입 ('data_collection', 'labeling', 'augmentation', 'validation')

    Returns:
        Mock: Mock 서비스 객체
    """
    mock_service = Mock()
    mock_service.initialize = Mock(return_value=True)
    mock_service.cleanup = Mock()
    mock_service.health_check = Mock(return_value=True)

    if service_type == "data_collection":
        mock_service.collect_files = Mock(return_value=["test1.pdf", "test2.pdf"])
        mock_service.get_collection_statistics = Mock(return_value={"total_files": 2})
    elif service_type == "labeling":
        mock_service.create_labeling_session = Mock(return_value="session_123")
        mock_service.get_labeling_progress = Mock(return_value={"progress": 0.5})
    elif service_type == "augmentation":
        mock_service.augment_dataset = Mock(return_value=[{"data": "augmented"}])
        mock_service.get_augmentation_statistics = Mock(
            return_value={"augmentation_factor": 3}
        )
    elif service_type == "validation":
        mock_service.validate_dataset = Mock(return_value={"validation_passed": True})
        mock_service.get_validation_statistics = Mock(
            return_value={"quality_score": 0.95}
        )

    return mock_service


# ====================================================================================
# 8. 테스트 성능 측정 함수들
# ====================================================================================


def measure_memory_usage(test_func: Callable) -> Dict[str, float]:
    """
    테스트 메모리 사용량 측정

    Args:
        test_func: 테스트 함수

    Returns:
        Dict[str, float]: 메모리 사용량 정보
    """
    try:
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # 시작 메모리 사용량
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 테스트 실행
        test_func()

        # 최종 메모리 사용량
        final_memory = process.memory_info().rss / 1024 / 1024  # MB

        return {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": final_memory,
            "memory_increase_mb": final_memory - initial_memory,
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def measure_cpu_usage(test_func: Callable) -> Dict[str, float]:
    """
    테스트 CPU 사용량 측정

    Args:
        test_func: 테스트 함수

    Returns:
        Dict[str, float]: CPU 사용량 정보
    """
    try:
        import psutil
        import time

        # CPU 사용률 측정 시작
        psutil.cpu_percent(interval=None)
        start_time = time.time()

        # 테스트 실행
        test_func()

        # 측정 완료
        end_time = time.time()
        cpu_usage = psutil.cpu_percent(interval=None)

        return {
            "execution_time_seconds": end_time - start_time,
            "cpu_usage_percent": cpu_usage,
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


# ====================================================================================
# 9. 테스트 디버깅 함수들
# ====================================================================================


def debug_test_failure(
    test_name: str, exception: Exception, context: Dict[str, Any]
) -> None:
    """
    테스트 실패 디버깅 정보 출력

    Args:
        test_name: 테스트 이름
        exception: 발생한 예외
        context: 테스트 컨텍스트
    """
    print(f"\n{'='*60}")
    print(f"테스트 실패 디버깅 정보: {test_name}")
    print(f"{'='*60}")
    print(f"예외 타입: {type(exception).__name__}")
    print(f"예외 메시지: {str(exception)}")
    print(f"테스트 컨텍스트:")
    for key, value in context.items():
        print(f"  {key}: {value}")

    # 스택 트레이스 출력
    import traceback

    print(f"\n스택 트레이스:")
    traceback.print_exc()
    print(f"{'='*60}\n")


def capture_test_output(test_func: Callable) -> tuple:
    """
    테스트 출력 캡처

    Args:
        test_func: 테스트 함수

    Returns:
        tuple: (stdout, stderr, 테스트 결과)
    """
    import io
    import sys

    # 출력 캡처용 버퍼
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # 기존 출력 백업
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    try:
        # 출력 리다이렉트
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer

        # 테스트 실행
        result = test_func()

        # 출력 내용 반환
        return stdout_buffer.getvalue(), stderr_buffer.getvalue(), result

    finally:
        # 원래 출력 복원
        sys.stdout = original_stdout
        sys.stderr = original_stderr


# ====================================================================================
# 10. 테스트 패키지 초기화 함수
# ====================================================================================


def initialize_test_package() -> bool:
    """
    테스트 패키지 초기화

    Returns:
        bool: 초기화 성공 여부
    """
    try:
        # 테스트 환경 검증
        if not validate_test_environment():
            print("⚠️ 테스트 환경 검증 실패")
            return False

        # 테스트 리포트 디렉터리 생성
        os.makedirs(TEST_REPORT_DIR, exist_ok=True)

        # 테스트 로거 설정
        test_logger = create_test_logger("test_package")
        test_logger.info("테스트 패키지 초기화 완료")

        return True
    except Exception as e:
        print(f"❌ 테스트 패키지 초기화 실패: {e}")
        return False


# ====================================================================================
# 11. 공개 API 정의
# ====================================================================================

__all__ = [
    # 메타데이터
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    # 테스트 설정
    "TEST_ENVIRONMENT",
    "TEST_LOG_LEVEL",
    "TEST_TIMEOUT_SECONDS",
    "TEST_BATCH_SIZE",
    "TEST_MAX_WORKERS",
    # 테스트 헬퍼 함수
    "create_test_config",
    "create_test_directories",
    "cleanup_test_directories",
    "create_test_logger",
    # 테스트 데이터 생성
    "create_test_pdf_file",
    "create_test_image_file",
    "create_test_json_file",
    "create_test_document_model",
    "create_test_annotation_model",
    "create_test_bounding_box",
    "create_temp_test_file",
    # 테스트 데이터 검증
    "validate_test_environment",
    "validate_test_data_integrity",
    "compare_test_results",
    # 테스트 실행 유틸리티
    "run_test_suite",
    "measure_test_execution_time",
    "create_test_report",
    # 목 객체 생성
    "create_mock_config",
    "create_mock_logger",
    "create_mock_file_handler",
    "create_mock_service",
    # 성능 측정
    "measure_memory_usage",
    "measure_cpu_usage",
    # 디버깅
    "debug_test_failure",
    "capture_test_output",
    # 초기화
    "initialize_test_package",
]

# ====================================================================================
# 12. 패키지 자동 초기화
# ====================================================================================

# 패키지 로드 시 자동 초기화
_initialization_success = initialize_test_package()

if _initialization_success:
    print("✅ YOKOGAWA OCR 테스트 패키지 초기화 완료")
else:
    print("❌ YOKOGAWA OCR 테스트 패키지 초기화 실패")

# 개발 환경에서만 패키지 정보 출력
if os.getenv("YOKOGAWA_OCR_DEBUG", "false").lower() == "true":
    print(f"📋 테스트 패키지 v{__version__} 로드됨")
    print(f"🔧 테스트 환경: {TEST_ENVIRONMENT}")
    print(f"📊 사용 가능한 헬퍼 함수: {len(__all__)}개")
