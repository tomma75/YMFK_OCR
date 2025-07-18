#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 데이터 수집 서비스 모듈

이 모듈은 PDF 및 이미지 파일을 수집하고 검증하는 서비스를 제공합니다.
파일 시스템에서 지원되는 형식의 파일들을 수집하고 메타데이터를 추출합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from core.base_classes import BaseService, DataCollectionInterface
from core.exceptions import (
    DataCollectionError,
    FileAccessError,
    FileFormatError,
    ProcessingError,
    ValidationError,
)
from config.settings import ApplicationConfig
from config.constants import (
    SUPPORTED_FILE_FORMATS,
    PDF_FILE_EXTENSIONS,
    IMAGE_FILE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    DEFAULT_BATCH_SIZE,
    FILE_PROCESSING_TIMEOUT_SECONDS,
)
from models.document_model import DocumentModel, DocumentType
from utils.file_handler import FileHandler
from utils.logger_util import get_application_logger


class FileCollector:
    """
    파일 수집기 클래스

    지정된 경로에서 지원되는 파일 형식의 파일들을 수집합니다.
    """

    def __init__(self, supported_formats: List[str]):
        """
        FileCollector 초기화

        Args:
            supported_formats: 지원되는 파일 형식 목록
        """
        self.supported_formats = supported_formats
        self.logger = get_application_logger("file_collector")

    def collect_files_from_directory(
        self, source_path: str, recursive: bool = True
    ) -> List[str]:
        """
        디렉터리에서 파일 수집

        Args:
            source_path: 수집할 디렉터리 경로
            recursive: 하위 디렉터리 포함 여부

        Returns:
            List[str]: 수집된 파일 경로 목록

        Raises:
            FileAccessError: 디렉터리 접근 실패 시
        """
        try:
            if not os.path.exists(source_path):
                raise FileAccessError(
                    message=f"Source directory not found: {source_path}",
                    file_path=source_path,
                    access_type="read",
                )

            collected_files = []
            source_path_obj = Path(source_path)

            if recursive:
                pattern = "**/*"
                file_paths = source_path_obj.glob(pattern)
            else:
                pattern = "*"
                file_paths = source_path_obj.glob(pattern)

            for file_path in file_paths:
                if file_path.is_file():
                    file_extension = file_path.suffix.lower()
                    if file_extension in self.supported_formats:
                        collected_files.append(str(file_path.absolute()))

            self.logger.info(
                f"Collected {len(collected_files)} files from {source_path}"
            )
            return collected_files

        except Exception as e:
            self.logger.error(f"Failed to collect files from {source_path}: {str(e)}")
            raise DataCollectionError(
                message=f"File collection failed: {str(e)}",
                source_path=source_path,
                original_exception=e,
            )


class MetadataExtractor:
    """
    메타데이터 추출기 클래스

    수집된 파일들의 메타데이터를 추출합니다.
    """

    def __init__(self, file_handler: FileHandler):
        """
        MetadataExtractor 초기화

        Args:
            file_handler: 파일 처리 핸들러
        """
        self.file_handler = file_handler
        self.logger = get_application_logger("metadata_extractor")

    def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        파일 메타데이터 추출

        Args:
            file_path: 파일 경로

        Returns:
            Dict[str, Any]: 추출된 메타데이터

        Raises:
            ProcessingError: 메타데이터 추출 실패 시
        """
        try:
            metadata = self.file_handler.get_file_metadata(file_path)

            # 파일 타입별 추가 메타데이터 추출
            file_extension = Path(file_path).suffix.lower()

            if file_extension in PDF_FILE_EXTENSIONS:
                metadata["document_type"] = DocumentType.PDF.value
            elif file_extension in IMAGE_FILE_EXTENSIONS:
                metadata["document_type"] = DocumentType.IMAGE.value
            else:
                metadata["document_type"] = DocumentType.UNKNOWN.value

            # 추가 품질 정보
            metadata["is_valid"] = self._validate_file_quality(file_path, metadata)
            metadata["extraction_timestamp"] = datetime.now().isoformat()

            return metadata

        except Exception as e:
            self.logger.error(f"Failed to extract metadata from {file_path}: {str(e)}")
            raise ProcessingError(
                message=f"Metadata extraction failed: {str(e)}",
                processor_id="metadata_extractor",
                processing_stage="metadata_extraction",
                original_exception=e,
            )

    def _validate_file_quality(self, file_path: str, metadata: Dict[str, Any]) -> bool:
        """
        파일 품질 검증

        Args:
            file_path: 파일 경로
            metadata: 메타데이터

        Returns:
            bool: 품질 검증 결과
        """
        try:
            # 파일 크기 검증
            if metadata.get("file_size_mb", 0) > MAX_FILE_SIZE_MB:
                return False

            # 파일 무결성 검증
            return self.file_handler.validate_file_integrity(file_path)

        except Exception as e:
            self.logger.warning(
                f"File quality validation failed for {file_path}: {str(e)}"
            )
            return False


class DuplicateDetector:
    """
    중복 파일 탐지기 클래스

    수집된 파일들 중 중복 파일을 탐지합니다.
    """

    def __init__(self, file_handler: FileHandler):
        """
        DuplicateDetector 초기화

        Args:
            file_handler: 파일 처리 핸들러
        """
        self.file_handler = file_handler
        self.logger = get_application_logger("duplicate_detector")
        self._hash_cache: Dict[str, str] = {}

    def detect_duplicates(self, file_list: List[str]) -> List[str]:
        """
        중복 파일 탐지

        Args:
            file_list: 검사할 파일 목록

        Returns:
            List[str]: 고유한 파일 목록 (중복 제거됨)
        """
        try:
            unique_files = []
            file_hashes: Dict[str, str] = {}

            for file_path in file_list:
                try:
                    # 파일 해시 계산
                    file_hash = self._get_file_hash(file_path)

                    if file_hash not in file_hashes:
                        file_hashes[file_hash] = file_path
                        unique_files.append(file_path)
                    else:
                        self.logger.info(
                            f"Duplicate file detected: {file_path} (original: {file_hashes[file_hash]})"
                        )

                except Exception as e:
                    self.logger.warning(f"Failed to process file {file_path}: {str(e)}")
                    continue

            duplicate_count = len(file_list) - len(unique_files)
            self.logger.info(
                f"Removed {duplicate_count} duplicate files from {len(file_list)} total files"
            )

            return unique_files

        except Exception as e:
            self.logger.error(f"Duplicate detection failed: {str(e)}")
            raise ProcessingError(
                message=f"Duplicate detection failed: {str(e)}",
                processor_id="duplicate_detector",
                processing_stage="duplicate_detection",
                original_exception=e,
            )

    def _get_file_hash(self, file_path: str) -> str:
        """
        파일 해시 조회 (캐시 활용)

        Args:
            file_path: 파일 경로

        Returns:
            str: 파일 해시
        """
        if file_path in self._hash_cache:
            return self._hash_cache[file_path]

        file_hash = self.file_handler.calculate_file_hash(file_path)
        self._hash_cache[file_path] = file_hash
        return file_hash


class DataCollectionService(BaseService, DataCollectionInterface):
    """
    데이터 수집 서비스 클래스

    파일 수집, 메타데이터 추출, 중복 제거 등의 데이터 수집 기능을 제공합니다.
    BaseService와 DataCollectionInterface를 구현합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        DataCollectionService 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)

        # 컴포넌트 초기화
        self.file_handler = FileHandler(config)
        self.file_collector = FileCollector(SUPPORTED_FILE_FORMATS)
        self.metadata_extractor = MetadataExtractor(self.file_handler)
        self.duplicate_detector = DuplicateDetector(self.file_handler)

        # 설정 정보
        self.batch_size = config.processing_config.batch_size
        self.max_workers = config.processing_config.max_workers
        self.processing_timeout = FILE_PROCESSING_TIMEOUT_SECONDS

        # 상태 관리
        self.collected_files: List[str] = []
        self.collected_documents: List[DocumentModel] = []
        self.collection_statistics: Dict[str, Any] = {}
        self.collection_callbacks: List[Callable] = []

        # 진행 상태
        self.collection_progress: float = 0.0
        self.current_operation: Optional[str] = None
        self.processing_errors: List[str] = []

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """
        서비스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("Initializing DataCollectionService")

            # 상태 초기화
            with self._lock:
                self.collected_files.clear()
                self.collected_documents.clear()
                self.collection_statistics.clear()
                self.collection_callbacks.clear()
                self.processing_errors.clear()
                self.collection_progress = 0.0
                self.current_operation = None

            # 파일 핸들러 초기화 검증
            if not self.file_handler:
                raise ProcessingError("FileHandler initialization failed")

            self.logger.info("DataCollectionService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize DataCollectionService: {str(e)}")
            self._is_initialized = False
            return False

    def cleanup(self) -> None:
        """
        서비스 정리
        """
        try:
            self.logger.info("Cleaning up DataCollectionService")

            with self._lock:
                self.collected_files.clear()
                self.collected_documents.clear()
                self.collection_statistics.clear()
                self.collection_callbacks.clear()
                self.processing_errors.clear()

            # 파일 핸들러 정리
            if hasattr(self.file_handler, "cleanup_temp_files"):
                self.file_handler.cleanup_temp_files()

            self.logger.info("DataCollectionService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during DataCollectionService cleanup: {str(e)}")
    
    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 초기화 상태 확인
            if not self.is_initialized():
                self.logger.warning("Service not initialized")
                return False

            # 기본 컴포넌트 확인
            if not hasattr(self, "config") or self.config is None:
                self.logger.warning("Config is None")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False

    def collect_files(self, source_path: str) -> List[str]:
        """
        파일 수집 (DataCollectionInterface 구현)

        Args:
            source_path: 수집할 경로

        Returns:
            List[str]: 수집된 파일 경로 목록
        """
        try:
            self.logger.info(f"Starting file collection from: {source_path}")

            with self._lock:
                self.current_operation = "file_collection"
                self.collection_progress = 0.0

            # 파일 수집
            self._update_progress(0.1, "Collecting files from directory")
            raw_files = self.file_collector.collect_files_from_directory(
                source_path, recursive=True
            )

            # 중복 제거
            self._update_progress(0.3, "Removing duplicate files")
            unique_files = self.duplicate_detector.detect_duplicates(raw_files)

            # 파일 검증
            self._update_progress(0.5, "Validating file integrity")
            validated_files = self._validate_files_batch(unique_files)

            # 메타데이터 추출 및 문서 모델 생성
            self._update_progress(
                0.7, "Extracting metadata and creating document models"
            )
            document_models = self._create_document_models(validated_files)

            # 결과 저장
            with self._lock:
                self.collected_files = validated_files
                self.collected_documents = document_models
                self.collection_progress = 1.0
                self.current_operation = None

            # 통계 업데이트
            self._update_collection_statistics()

            # 콜백 실행
            self._execute_collection_callbacks()

            self.logger.info(
                f"File collection completed: {len(validated_files)} files collected"
            )
            return validated_files

        except Exception as e:
            self.logger.error(f"File collection failed: {str(e)}")
            with self._lock:
                self.processing_errors.append(str(e))
                self.current_operation = None
            raise DataCollectionError(
                message=f"File collection failed: {str(e)}",
                source_path=source_path,
                original_exception=e,
            )

    def collect_pdf_files(self, source_path: str) -> List[str]:
        """
        PDF 파일 수집

        Args:
            source_path: 수집할 경로

        Returns:
            List[str]: 수집된 PDF 파일 경로 목록
        """
        try:
            all_files = self.collect_files(source_path)
            pdf_files = [
                f for f in all_files if Path(f).suffix.lower() in PDF_FILE_EXTENSIONS
            ]

            self.logger.info(f"Collected {len(pdf_files)} PDF files from {source_path}")
            return pdf_files

        except Exception as e:
            self.logger.error(f"PDF file collection failed: {str(e)}")
            raise DataCollectionError(
                message=f"PDF file collection failed: {str(e)}",
                source_path=source_path,
                original_exception=e,
            )

    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        수집 통계 정보 제공 (DataCollectionInterface 구현)

        Returns:
            Dict[str, Any]: 수집 통계 정보
        """
        with self._lock:
            return self.collection_statistics.copy()

    def register_collection_callback(self, callback: Callable) -> None:
        """
        수집 완료 시 콜백 등록 (DataCollectionInterface 구현)

        Args:
            callback: 콜백 함수
        """
        with self._lock:
            self.collection_callbacks.append(callback)

        self.logger.debug(f"Collection callback registered: {callback.__name__}")

    def categorize_files_by_type(self, file_list: List[str]) -> Dict[str, List[str]]:
        """
        파일 타입별 분류

        Args:
            file_list: 분류할 파일 목록

        Returns:
            Dict[str, List[str]]: 타입별 분류된 파일 목록
        """
        try:
            categorized_files = {"pdf": [], "image": [], "unknown": []}

            for file_path in file_list:
                file_extension = Path(file_path).suffix.lower()

                if file_extension in PDF_FILE_EXTENSIONS:
                    categorized_files["pdf"].append(file_path)
                elif file_extension in IMAGE_FILE_EXTENSIONS:
                    categorized_files["image"].append(file_path)
                else:
                    categorized_files["unknown"].append(file_path)

            self.logger.info(
                f"Files categorized: PDF={len(categorized_files['pdf'])}, "
                f"Image={len(categorized_files['image'])}, "
                f"Unknown={len(categorized_files['unknown'])}"
            )

            return categorized_files

        except Exception as e:
            self.logger.error(f"File categorization failed: {str(e)}")
            raise ProcessingError(
                message=f"File categorization failed: {str(e)}",
                processor_id=self.service_id,
                processing_stage="file_categorization",
                original_exception=e,
            )

    def validate_file_integrity(self, file_path: str) -> bool:
        """
        파일 무결성 검증

        Args:
            file_path: 검증할 파일 경로

        Returns:
            bool: 무결성 검증 결과
        """
        try:
            return self.file_handler.validate_file_integrity(file_path)

        except Exception as e:
            self.logger.error(
                f"File integrity validation failed for {file_path}: {str(e)}"
            )
            return False

    def extract_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        파일 메타데이터 추출

        Args:
            file_path: 파일 경로

        Returns:
            Dict[str, Any]: 추출된 메타데이터
        """
        try:
            return self.metadata_extractor.extract_file_metadata(file_path)

        except Exception as e:
            self.logger.error(f"Metadata extraction failed for {file_path}: {str(e)}")
            raise ProcessingError(
                message=f"Metadata extraction failed: {str(e)}",
                processor_id=self.service_id,
                processing_stage="metadata_extraction",
                original_exception=e,
            )

    def detect_duplicates(self, file_list: List[str]) -> List[str]:
        """
        중복 파일 탐지

        Args:
            file_list: 검사할 파일 목록

        Returns:
            List[str]: 고유한 파일 목록
        """
        try:
            return self.duplicate_detector.detect_duplicates(file_list)

        except Exception as e:
            self.logger.error(f"Duplicate detection failed: {str(e)}")
            raise ProcessingError(
                message=f"Duplicate detection failed: {str(e)}",
                processor_id=self.service_id,
                processing_stage="duplicate_detection",
                original_exception=e,
            )

    def get_collected_documents(self) -> List[DocumentModel]:
        """
        수집된 문서 모델 목록 반환

        Returns:
            List[DocumentModel]: 수집된 문서 모델 목록
        """
        with self._lock:
            return self.collected_documents.copy()

    def get_collection_progress(self) -> Dict[str, Any]:
        """
        수집 진행 상황 반환

        Returns:
            Dict[str, Any]: 진행 상황 정보
        """
        with self._lock:
            return {
                "progress": self.collection_progress,
                "current_operation": self.current_operation,
                "collected_files_count": len(self.collected_files),
                "collected_documents_count": len(self.collected_documents),
                "processing_errors_count": len(self.processing_errors),
            }

    def _validate_files_batch(self, file_list: List[str]) -> List[str]:
        """
        파일 배치 검증

        Args:
            file_list: 검증할 파일 목록

        Returns:
            List[str]: 검증 통과 파일 목록
        """
        validated_files = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.validate_file_integrity, file_path): file_path
                for file_path in file_list
            }

            for future in as_completed(future_to_file, timeout=self.processing_timeout):
                file_path = future_to_file[future]
                try:
                    is_valid = future.result()
                    if is_valid:
                        validated_files.append(file_path)
                    else:
                        self.logger.warning(f"File validation failed: {file_path}")
                except Exception as e:
                    self.logger.error(f"Error validating file {file_path}: {str(e)}")

        return validated_files

    def _create_document_models(self, file_list: List[str]) -> List[DocumentModel]:
        """
        문서 모델 생성

        Args:
            file_list: 파일 목록

        Returns:
            List[DocumentModel]: 생성된 문서 모델 목록
        """
        document_models = []

        for file_path in file_list:
            try:
                document_model = DocumentModel.from_file_path(file_path)
                document_models.append(document_model)
            except Exception as e:
                self.logger.error(
                    f"Failed to create document model for {file_path}: {str(e)}"
                )
                with self._lock:
                    self.processing_errors.append(
                        f"Document model creation failed for {file_path}: {str(e)}"
                    )

        return document_models

    def _update_progress(self, progress: float, operation: str) -> None:
        """
        진행 상황 업데이트

        Args:
            progress: 진행률 (0.0 ~ 1.0)
            operation: 현재 작업 설명
        """
        with self._lock:
            self.collection_progress = progress
            self.current_operation = operation

        self.logger.debug(f"Progress updated: {progress:.1%} - {operation}")

    def _update_collection_statistics(self) -> None:
        """
        수집 통계 업데이트
        """
        with self._lock:
            file_types = self.categorize_files_by_type(self.collected_files)

            self.collection_statistics = {
                "total_files_collected": len(self.collected_files),
                "total_documents_created": len(self.collected_documents),
                "file_types": {
                    "pdf_count": len(file_types["pdf"]),
                    "image_count": len(file_types["image"]),
                    "unknown_count": len(file_types["unknown"]),
                },
                "processing_errors_count": len(self.processing_errors),
                "collection_timestamp": datetime.now().isoformat(),
                "service_id": self.service_id,
            }

    def _execute_collection_callbacks(self) -> None:
        """
        수집 완료 콜백 실행
        """
        with self._lock:
            callbacks = self.collection_callbacks.copy()

        for callback in callbacks:
            try:
                callback(self.collected_documents)
            except Exception as e:
                self.logger.error(f"Collection callback execution failed: {str(e)}")

    @classmethod
    def create_with_dependencies(cls, container) -> "DataCollectionService":
        """
        의존성 컨테이너를 사용한 팩토리 메서드

        Args:
            container: 의존성 컨테이너

        Returns:
            DataCollectionService: 생성된 서비스 인스턴스
        """
        return cls(
            config=container.get_service("config"),
            logger=container.get_service("logger"),
        )


# 모듈 수준 유틸리티 함수들
def create_data_collection_service(config: ApplicationConfig) -> DataCollectionService:
    """
    데이터 수집 서비스 생성 함수

    Args:
        config: 애플리케이션 설정

    Returns:
        DataCollectionService: 생성된 서비스 인스턴스
    """
    logger = get_application_logger("data_collection_service")
    service = DataCollectionService(config, logger)

    if not service.initialize():
        raise ProcessingError("Failed to initialize DataCollectionService")

    return service


if __name__ == "__main__":
    # 데이터 수집 서비스 테스트
    print("YOKOGAWA OCR 데이터 수집 서비스 테스트")
    print("=" * 50)

    try:
        # 설정 로드
        from config.settings import load_configuration

        config = load_configuration()

        # 서비스 생성
        service = create_data_collection_service(config)

        # 상태 확인
        if service.health_check():
            print("✅ 데이터 수집 서비스 정상 동작")
        else:
            print("❌ 데이터 수집 서비스 상태 이상")

        # 통계 정보 출력
        statistics = service.get_collection_statistics()
        print(f"📊 수집 통계: {statistics}")

        # 정리
        service.cleanup()

    except Exception as e:
        print(f"❌ 데이터 수집 서비스 테스트 실패: {e}")

    print("\n🎯 데이터 수집 서비스 구현이 완료되었습니다!")
