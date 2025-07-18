#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 시스템 상수 정의 모듈

이 모듈은 전체 시스템에서 사용되는 상수들을 정의하며,
매직 넘버를 제거하고 설정값의 일관성을 보장합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import logging
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path

# ====================================================================================
# 1. 파일 처리 관련 상수
# ====================================================================================

# 지원되는 파일 형식
SUPPORTED_FILE_FORMATS: List[str] = [
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
]

# PDF 관련 상수
PDF_FILE_EXTENSIONS: List[str] = [".pdf"]
PDF_MAX_PAGE_COUNT: int = 1000
PDF_DEFAULT_DPI: int = 300
PDF_EXTRACTION_TIMEOUT_SECONDS: int = 300

# 이미지 관련 상수
IMAGE_FILE_EXTENSIONS: List[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".tiff",
    ".tif",
    ".bmp",
    ".gif",
]
IMAGE_MAX_DIMENSION: int = 8192
IMAGE_MIN_DIMENSION: int = 64
IMAGE_DEFAULT_QUALITY: int = 95

# 파일 크기 제한 (MB 단위)
MAX_FILE_SIZE_MB: int = 500
MAX_PDF_FILE_SIZE_MB: int = 200
MAX_IMAGE_FILE_SIZE_MB: int = 100
MIN_FILE_SIZE_KB: int = 1

# 파일 처리 배치 크기
DEFAULT_BATCH_SIZE: int = 10
MAX_BATCH_SIZE: int = 50
MIN_BATCH_SIZE: int = 1

# 파일 처리 타임아웃
FILE_PROCESSING_TIMEOUT_SECONDS: int = 300
FILE_COPY_TIMEOUT_SECONDS: int = 60
FILE_VALIDATION_TIMEOUT_SECONDS: int = 30

# ====================================================================================
# 2. 이미지 처리 관련 상수
# ====================================================================================

# 기본 이미지 해상도
DEFAULT_IMAGE_RESOLUTION: int = 300
HIGH_RESOLUTION_DPI: int = 600
LOW_RESOLUTION_DPI: int = 150

# 이미지 변환 설정
IMAGE_CONVERSION_QUALITY: int = 95
IMAGE_COMPRESSION_LEVEL: int = 6
IMAGE_INTERPOLATION_METHOD: str = "LANCZOS"

# 이미지 전처리 설정
IMAGE_NOISE_REDUCTION_KERNEL_SIZE: int = 3
IMAGE_BLUR_KERNEL_SIZE: int = 5
IMAGE_SHARPENING_STRENGTH: float = 0.5

# 이미지 증강 파라미터
AUGMENTATION_ROTATION_RANGE: Tuple[int, int] = (-10, 10)
AUGMENTATION_SCALE_RANGE: Tuple[float, float] = (0.9, 1.1)
AUGMENTATION_BRIGHTNESS_RANGE: Tuple[float, float] = (0.8, 1.2)
AUGMENTATION_CONTRAST_RANGE: Tuple[float, float] = (0.8, 1.2)
AUGMENTATION_NOISE_VARIANCE: float = 0.01

# 이미지 색상 공간
DEFAULT_COLOR_SPACE: str = "RGB"
GRAYSCALE_COLOR_SPACE: str = "L"
CMYK_COLOR_SPACE: str = "CMYK"

# ====================================================================================
# 3. 데이터 분할 관련 상수
# ====================================================================================

# 데이터 분할 비율
DATA_SPLIT_RATIOS: Dict[str, float] = {"train": 0.7, "validation": 0.2, "test": 0.1}

# 최소 데이터 분할 크기
MIN_TRAIN_DATASET_SIZE: int = 100
MIN_VALIDATION_DATASET_SIZE: int = 20
MIN_TEST_DATASET_SIZE: int = 10

# 데이터 분할 전략
DATA_SPLIT_STRATEGY: str = "stratified"
DATA_SPLIT_RANDOM_STATE: int = 42
DATA_SPLIT_SHUFFLE: bool = True

# 크로스 검증 설정
CROSS_VALIDATION_FOLDS: int = 5
CROSS_VALIDATION_RANDOM_STATE: int = 42

# ====================================================================================
# 4. 어노테이션 관련 상수
# ====================================================================================

# 어노테이션 필드 타입
ANNOTATION_FIELD_TYPES: Dict[str, str] = {
    "text": "텍스트 입력 필드",
    "number": "숫자 입력 필드",
    "date": "날짜 입력 필드",
    "checkbox": "체크박스 필드",
    "dropdown": "드롭다운 선택 필드",
    "signature": "서명 필드",
    "table": "테이블 필드",
    "image": "이미지 필드",
}

# 어노테이션 검증 규칙
ANNOTATION_MIN_TEXT_LENGTH: int = 1
ANNOTATION_MAX_TEXT_LENGTH: int = 1000
ANNOTATION_MIN_BOUNDING_BOX_SIZE: int = 5
ANNOTATION_MAX_BOUNDING_BOX_SIZE: int = 2000

# 어노테이션 품질 임계값
ANNOTATION_QUALITY_THRESHOLD: float = 0.8
ANNOTATION_CONFIDENCE_THRESHOLD: float = 0.7
ANNOTATION_COMPLETENESS_THRESHOLD: float = 0.9

# 어노테이션 세션 설정
ANNOTATION_SESSION_TIMEOUT_MINUTES: int = 60
ANNOTATION_AUTO_SAVE_INTERVAL_SECONDS: int = 30
ANNOTATION_MAX_ACTIVE_SESSIONS: int = 10

# 바운딩 박스 제약 조건
BOUNDING_BOX_MIN_WIDTH: int = 10
BOUNDING_BOX_MIN_HEIGHT: int = 10
BOUNDING_BOX_MAX_ASPECT_RATIO: float = 20.0
BOUNDING_BOX_MIN_ASPECT_RATIO: float = 0.05

# ====================================================================================
# 5. 데이터 증강 관련 상수
# ====================================================================================

# 증강 배수
DEFAULT_AUGMENTATION_FACTOR: int = 3
MAX_AUGMENTATION_FACTOR: int = 10
MIN_AUGMENTATION_FACTOR: int = 1

# 기하학적 변환 파라미터
GEOMETRIC_ROTATION_ANGLES: List[float] = [-5.0, -2.0, 2.0, 5.0]
GEOMETRIC_SCALE_FACTORS: List[float] = [0.95, 1.05, 1.1]
GEOMETRIC_TRANSLATION_RANGE: int = 10
GEOMETRIC_SHEAR_RANGE: float = 0.1

# 색상 변환 파라미터
COLOR_BRIGHTNESS_DELTA: float = 0.2
COLOR_CONTRAST_DELTA: float = 0.2
COLOR_SATURATION_DELTA: float = 0.2
COLOR_HUE_DELTA: float = 0.1

# 노이즈 추가 파라미터
NOISE_GAUSSIAN_MEAN: float = 0.0
NOISE_GAUSSIAN_STD: float = 0.05
NOISE_SALT_PEPPER_AMOUNT: float = 0.01
NOISE_SPECKLE_VARIANCE: float = 0.1

# ====================================================================================
# 6. 검증 관련 상수
# ====================================================================================

# 데이터 품질 검증 임계값
DATA_QUALITY_MIN_SCORE: float = 0.8
DATA_COMPLETENESS_MIN_SCORE: float = 0.9
DATA_CONSISTENCY_MIN_SCORE: float = 0.85

# 검증 규칙 가중치
VALIDATION_COMPLETENESS_WEIGHT: float = 0.4
VALIDATION_ACCURACY_WEIGHT: float = 0.3
VALIDATION_CONSISTENCY_WEIGHT: float = 0.3

# 검증 시도 횟수
VALIDATION_MAX_RETRY_COUNT: int = 3
VALIDATION_RETRY_DELAY_SECONDS: int = 1

# 통계 계산 설정
STATISTICS_CONFIDENCE_LEVEL: float = 0.95
STATISTICS_SAMPLE_SIZE: int = 1000
STATISTICS_PRECISION_DIGITS: int = 4

# ====================================================================================
# 7. 시스템 성능 관련 상수
# ====================================================================================

# 병렬 처리 설정
DEFAULT_MAX_WORKERS: int = 4
MAX_WORKER_THREADS: int = 16
MIN_WORKER_THREADS: int = 1

# 메모리 사용 제한
MAX_MEMORY_USAGE_MB: int = 8192
MEMORY_WARNING_THRESHOLD_MB: int = 6144
MEMORY_CRITICAL_THRESHOLD_MB: int = 7680

# 처리 타임아웃 설정
PROCESSING_TIMEOUT_SECONDS: int = 300
NETWORK_TIMEOUT_SECONDS: int = 30
DATABASE_TIMEOUT_SECONDS: int = 60

# 캐시 설정
CACHE_SIZE_LIMIT_MB: int = 1024
CACHE_EXPIRY_SECONDS: int = 3600
CACHE_CLEANUP_INTERVAL_SECONDS: int = 600

# ====================================================================================
# 8. 로깅 관련 상수
# ====================================================================================

# 로그 레벨
LOG_LEVEL_DEBUG: str = "DEBUG"
LOG_LEVEL_INFO: str = "INFO"
LOG_LEVEL_WARNING: str = "WARNING"
LOG_LEVEL_ERROR: str = "ERROR"
LOG_LEVEL_CRITICAL: str = "CRITICAL"

# 로그 파일 설정
LOG_FILE_MAX_SIZE_MB: int = 100
LOG_FILE_BACKUP_COUNT: int = 5
LOG_FILE_ENCODING: str = "utf-8"

# 로그 포맷 설정
LOG_FORMAT_TIMESTAMP: str = "%Y-%m-%d %H:%M:%S"
LOG_FORMAT_TEMPLATE: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# 로그 로테이션 설정
LOG_ROTATION_WHEN: str = "midnight"
LOG_ROTATION_INTERVAL: int = 1
LOG_ROTATION_UTC: bool = False

# ====================================================================================
# 9. 디렉터리 및 파일 경로 상수
# ====================================================================================

# 기본 디렉터리 이름
DATA_DIRECTORY_NAME: str = "data"
RAW_DATA_DIRECTORY_NAME: str = "raw"
PROCESSED_DATA_DIRECTORY_NAME: str = "processed"
ANNOTATIONS_DIRECTORY_NAME: str = "annotations"
AUGMENTED_DATA_DIRECTORY_NAME: str = "augmented"
TEMPLATES_DIRECTORY_NAME: str = "templates"
LOGS_DIRECTORY_NAME: str = "logs"
TEMP_DIRECTORY_NAME: str = "temp"

# 파일 확장자 매핑
FILE_EXTENSION_MAPPING: Dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
    ".gif": "image/gif",
}

# 백업 파일 설정
BACKUP_FILE_SUFFIX: str = ".backup"
TEMP_FILE_SUFFIX: str = ".tmp"
LOCK_FILE_SUFFIX: str = ".lock"

# ====================================================================================
# 10. 네트워크 및 API 관련 상수
# ====================================================================================

# HTTP 상태 코드
HTTP_STATUS_OK: int = 200
HTTP_STATUS_CREATED: int = 201
HTTP_STATUS_BAD_REQUEST: int = 400
HTTP_STATUS_UNAUTHORIZED: int = 401
HTTP_STATUS_FORBIDDEN: int = 403
HTTP_STATUS_NOT_FOUND: int = 404
HTTP_STATUS_INTERNAL_SERVER_ERROR: int = 500

# API 요청 설정
API_REQUEST_TIMEOUT_SECONDS: int = 30
API_MAX_RETRY_COUNT: int = 3
API_RETRY_DELAY_SECONDS: int = 1
API_RATE_LIMIT_REQUESTS_PER_MINUTE: int = 100

# 네트워크 버퍼 크기
NETWORK_BUFFER_SIZE_KB: int = 64
NETWORK_CHUNK_SIZE_KB: int = 8

# ====================================================================================
# 11. 데이터베이스 관련 상수
# ====================================================================================

# 데이터베이스 연결 설정
DATABASE_CONNECTION_POOL_SIZE: int = 20
DATABASE_CONNECTION_POOL_MIN_SIZE: int = 5
DATABASE_CONNECTION_TIMEOUT_SECONDS: int = 30
DATABASE_QUERY_TIMEOUT_SECONDS: int = 300

# 데이터베이스 페이징 설정
DATABASE_DEFAULT_PAGE_SIZE: int = 100
DATABASE_MAX_PAGE_SIZE: int = 1000
DATABASE_MIN_PAGE_SIZE: int = 10

# 트랜잭션 설정
DATABASE_TRANSACTION_TIMEOUT_SECONDS: int = 60
DATABASE_MAX_TRANSACTION_RETRY: int = 3

# ====================================================================================
# 12. 보안 관련 상수
# ====================================================================================

# 암호화 설정
ENCRYPTION_KEY_LENGTH: int = 32
ENCRYPTION_IV_LENGTH: int = 16
ENCRYPTION_ALGORITHM: str = "AES-256-CBC"

# 해시 설정
HASH_ALGORITHM: str = "SHA-256"
HASH_SALT_LENGTH: int = 32

# 세션 설정
SESSION_TIMEOUT_MINUTES: int = 60
SESSION_CLEANUP_INTERVAL_MINUTES: int = 10

# ====================================================================================
# 13. 텍스트 처리 관련 상수
# ====================================================================================

# 텍스트 인코딩
DEFAULT_TEXT_ENCODING: str = "utf-8"
FALLBACK_TEXT_ENCODING: str = "latin-1"

# 텍스트 정규화
TEXT_NORMALIZATION_FORM: str = "NFC"
TEXT_MIN_LENGTH: int = 1
TEXT_MAX_LENGTH: int = 10000

# 언어 설정
DEFAULT_LANGUAGE: str = "ko"
SUPPORTED_LANGUAGES: List[str] = ["ko", "en", "ja", "zh"]

# 텍스트 분할 설정
TEXT_CHUNK_SIZE: int = 1000
TEXT_CHUNK_OVERLAP: int = 100

# ====================================================================================
# 14. 품질 메트릭 관련 상수
# ====================================================================================

# 품질 점수 범위
QUALITY_SCORE_MIN: float = 0.0
QUALITY_SCORE_MAX: float = 1.0
QUALITY_SCORE_PRECISION: int = 3

# 품질 등급 임계값
QUALITY_GRADE_EXCELLENT: float = 0.9
QUALITY_GRADE_GOOD: float = 0.8
QUALITY_GRADE_FAIR: float = 0.7
QUALITY_GRADE_POOR: float = 0.6

# 성능 메트릭 설정
PERFORMANCE_METRIC_WINDOW_SIZE: int = 100
PERFORMANCE_METRIC_SAMPLE_INTERVAL: int = 5

# ====================================================================================
# 15. 환경 변수 키 상수
# ====================================================================================

# 애플리케이션 환경 변수
ENV_VAR_CONFIG_FILE: str = "YOKOGAWA_CONFIG_FILE"
ENV_VAR_ENVIRONMENT: str = "YOKOGAWA_ENVIRONMENT"
ENV_VAR_DEBUG_MODE: str = "YOKOGAWA_DEBUG_MODE"
ENV_VAR_LOG_LEVEL: str = "YOKOGAWA_LOG_LEVEL"

# 데이터베이스 환경 변수
ENV_VAR_DB_HOST: str = "YOKOGAWA_DB_HOST"
ENV_VAR_DB_PORT: str = "YOKOGAWA_DB_PORT"
ENV_VAR_DB_NAME: str = "YOKOGAWA_DB_NAME"
ENV_VAR_DB_USERNAME: str = "YOKOGAWA_DB_USERNAME"
ENV_VAR_DB_PASSWORD: str = "YOKOGAWA_DB_PASSWORD"

# 디렉터리 환경 변수
ENV_VAR_DATA_DIR: str = "YOKOGAWA_DATA_DIR"
ENV_VAR_LOG_DIR: str = "YOKOGAWA_LOG_DIR"
ENV_VAR_TEMP_DIR: str = "YOKOGAWA_TEMP_DIR"

# ====================================================================================
# 16. 유틸리티 함수들
# ====================================================================================


def get_supported_extensions() -> List[str]:
    """지원되는 파일 확장자 목록 반환"""
    return SUPPORTED_FILE_FORMATS.copy()


def is_supported_file_format(file_extension: str) -> bool:
    """
    파일 확장자가 지원되는지 확인

    Args:
        file_extension: 확인할 파일 확장자

    Returns:
        bool: 지원 여부
    """
    return file_extension.lower() in SUPPORTED_FILE_FORMATS


def get_file_type_by_extension(file_extension: str) -> str:
    """
    파일 확장자로 파일 타입 반환

    Args:
        file_extension: 파일 확장자

    Returns:
        str: 파일 타입
    """
    if file_extension.lower() in PDF_FILE_EXTENSIONS:
        return "pdf"
    elif file_extension.lower() in IMAGE_FILE_EXTENSIONS:
        return "image"
    else:
        return "unknown"


def get_max_file_size_by_type(file_type: str) -> int:
    """
    파일 타입별 최대 파일 크기 반환 (MB)

    Args:
        file_type: 파일 타입

    Returns:
        int: 최대 파일 크기 (MB)
    """
    if file_type == "pdf":
        return MAX_PDF_FILE_SIZE_MB
    elif file_type == "image":
        return MAX_IMAGE_FILE_SIZE_MB
    else:
        return MAX_FILE_SIZE_MB


def get_quality_grade(score: float) -> str:
    """
    품질 점수에 따른 등급 반환

    Args:
        score: 품질 점수 (0.0 ~ 1.0)

    Returns:
        str: 품질 등급
    """
    if score >= QUALITY_GRADE_EXCELLENT:
        return "excellent"
    elif score >= QUALITY_GRADE_GOOD:
        return "good"
    elif score >= QUALITY_GRADE_FAIR:
        return "fair"
    elif score >= QUALITY_GRADE_POOR:
        return "poor"
    else:
        return "unacceptable"


def validate_data_split_ratios(ratios: Dict[str, float]) -> bool:
    """
    데이터 분할 비율 검증

    Args:
        ratios: 분할 비율 딕셔너리

    Returns:
        bool: 검증 결과
    """
    total_ratio = sum(ratios.values())
    return abs(total_ratio - 1.0) < 0.001

def validate_environment_setup() -> bool:
    """
    환경 설정 검증
    
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 필수 환경변수 확인
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
                    # 디렉토리 생성
                    os.makedirs(default_values[var], exist_ok=True)
                    logging.info(f"환경변수 {var} 기본값 설정: {default_values[var]}")
            
            logging.warning(f"Missing environment variables set to default values: {missing_vars}")
        
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

def get_environment_variable(key: str, default: str = None) -> str:
    """
    환경 변수 값 조회

    Args:
        key: 환경 변수 키
        default: 기본값

    Returns:
        str: 환경 변수 값
    """
    return os.getenv(key, default)


def create_directory_path(*parts: str) -> str:
    """
    디렉터리 경로 생성

    Args:
        *parts: 경로 부분들

    Returns:
        str: 생성된 경로
    """
    return str(Path(*parts))


# ====================================================================================
# 17. 상수 검증 함수들
# ====================================================================================


def validate_constants() -> bool:
    """
    모든 상수들의 유효성 검증

    Returns:
        bool: 검증 결과
    """
    try:
        # 데이터 분할 비율 검증
        if not validate_data_split_ratios(DATA_SPLIT_RATIOS):
            return False

        # 파일 크기 제한 검증
        if MAX_FILE_SIZE_MB <= 0:
            return False

        # 이미지 해상도 검증
        if DEFAULT_IMAGE_RESOLUTION <= 0:
            return False

        # 워커 스레드 수 검증
        if DEFAULT_MAX_WORKERS <= 0:
            return False

        # 품질 임계값 검증
        if not (0.0 <= ANNOTATION_QUALITY_THRESHOLD <= 1.0):
            return False

        return True

    except Exception:
        return False


def get_constants_summary() -> Dict[str, Any]:
    """
    상수 요약 정보 반환

    Returns:
        Dict[str, Any]: 상수 요약 정보
    """
    return {
        "supported_file_formats": len(SUPPORTED_FILE_FORMATS),
        "annotation_field_types": len(ANNOTATION_FIELD_TYPES),
        "data_split_ratios": DATA_SPLIT_RATIOS,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
        "default_image_resolution": DEFAULT_IMAGE_RESOLUTION,
        "default_max_workers": DEFAULT_MAX_WORKERS,
        "validation_status": validate_constants(),
    }


# ====================================================================================
# 18. 런타임 상수 초기화
# ====================================================================================

# 모듈 로드 시 상수 검증 실행
if __name__ == "__main__":
    print("YOKOGAWA OCR 상수 검증 중...")

    if validate_constants():
        print("✅ 모든 상수가 유효합니다.")

        # 상수 요약 정보 출력
        summary = get_constants_summary()
        print(f"📊 상수 요약 정보:")
        for key, value in summary.items():
            print(f"  - {key}: {value}")

    else:
        print("❌ 상수 검증에 실패했습니다.")

    # 지원되는 파일 형식 출력
    print(f"\n📁 지원되는 파일 형식: {', '.join(SUPPORTED_FILE_FORMATS)}")

    # 데이터 분할 비율 출력
    print(f"📊 데이터 분할 비율: {DATA_SPLIT_RATIOS}")

    # 품질 임계값 출력
    print(f"🎯 품질 임계값: {ANNOTATION_QUALITY_THRESHOLD}")
