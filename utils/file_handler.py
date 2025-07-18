#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 파일 처리 유틸리티 모듈

이 모듈은 파일 시스템 작업, 파일 변환, 압축/해제 등의 파일 처리 기능을 제공합니다.
PDF, 이미지, JSON 파일 등 다양한 형식의 파일을 처리할 수 있습니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import shutil
import json
import hashlib
import zipfile
import tarfile
import gzip
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, BinaryIO
from datetime import datetime
import mimetypes
import stat

from core.base_classes import BaseProcessor
from core.exceptions import (
    FileProcessingError,
    FileAccessError,
    ProcessingError,
    ValidationError,
    ApplicationError
)
from config.constants import (
    SUPPORTED_FILE_FORMATS,
    PDF_FILE_EXTENSIONS,
    IMAGE_FILE_EXTENSIONS,
    MAX_FILE_SIZE_MB,
    FILE_PROCESSING_TIMEOUT_SECONDS,
    FILE_COPY_TIMEOUT_SECONDS,
    BACKUP_FILE_SUFFIX,
    TEMP_FILE_SUFFIX,
    LOCK_FILE_SUFFIX,
    NETWORK_BUFFER_SIZE_KB,
    NETWORK_CHUNK_SIZE_KB
)
from config.settings import ApplicationConfig
from utils.logger_util import get_application_logger


# ====================================================================================
# 1. 메인 파일 핸들러 클래스
# ====================================================================================

class FileHandler:
    """
    파일 처리 유틸리티 클래스
    
    파일 시스템 작업, 파일 복사, 백업 등의 기본적인 파일 처리 기능을 제공합니다.
    """
    
    def __init__(self, config: Optional[ApplicationConfig] = None):
        """
        FileHandler 초기화
        
        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger('file_handler')
        self.temp_directory = tempfile.gettempdir()
        self.buffer_size = NETWORK_BUFFER_SIZE_KB * 1024
        self.chunk_size = NETWORK_CHUNK_SIZE_KB * 1024
        
        # 지원되는 파일 형식 설정
        self.supported_formats = SUPPORTED_FILE_FORMATS
        
        # 처리 통계
        self.processed_files_count = 0
        self.failed_files_count = 0
        self.total_bytes_processed = 0
        
        self.logger.info("FileHandler initialized successfully")
    
    def create_directory_if_not_exists(self, directory_path: str) -> bool:
        """
        디렉터리가 존재하지 않으면 생성
        
        Args:
            directory_path: 생성할 디렉터리 경로
            
        Returns:
            bool: 생성 성공 여부
        """
        try:
            path = Path(directory_path)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.info(f"Directory created: {directory_path}")
            
            # 디렉터리 권한 확인
            if not os.access(directory_path, os.W_OK):
                raise FileAccessError(
                    message=f"No write permission for directory: {directory_path}",
                    file_path=directory_path,
                    access_type="write"
                )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create directory {directory_path}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to create directory: {str(e)}",
                file_path=directory_path,
                operation="create_directory",
                original_exception=e
            )
    
    def copy_file_with_backup(self, source: str, destination: str) -> bool:
        """
        파일 복사 (기존 파일이 있으면 백업)
        
        Args:
            source: 원본 파일 경로
            destination: 대상 파일 경로
            
        Returns:
            bool: 복사 성공 여부
        """
        try:
            # 원본 파일 존재 확인
            if not os.path.exists(source):
                raise FileAccessError(
                    message=f"Source file not found: {source}",
                    file_path=source,
                    access_type="read"
                )
            
            # 대상 디렉터리 생성
            destination_dir = os.path.dirname(destination)
            if destination_dir:
                self.create_directory_if_not_exists(destination_dir)
            
            # 기존 파일 백업
            if os.path.exists(destination):
                backup_path = destination + BACKUP_FILE_SUFFIX
                shutil.copy2(destination, backup_path)
                self.logger.info(f"Existing file backed up: {backup_path}")
            
            # 파일 복사
            shutil.copy2(source, destination)
            
            # 복사 검증
            if not self._verify_file_copy(source, destination):
                raise FileProcessingError(
                    message="File copy verification failed",
                    file_path=source,
                    operation="copy_verification"
                )
            
            self.processed_files_count += 1
            self.total_bytes_processed += os.path.getsize(source)
            
            self.logger.info(f"File copied successfully: {source} -> {destination}")
            return True
            
        except Exception as e:
            self.failed_files_count += 1
            self.logger.error(f"Failed to copy file {source} to {destination}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to copy file: {str(e)}",
                file_path=source,
                operation="copy_file",
                original_exception=e
            )
    
    def move_file_with_backup(self, source: str, destination: str) -> bool:
        """
        파일 이동 (기존 파일이 있으면 백업)
        
        Args:
            source: 원본 파일 경로
            destination: 대상 파일 경로
            
        Returns:
            bool: 이동 성공 여부
        """
        try:
            # 파일 복사 먼저 수행
            if self.copy_file_with_backup(source, destination):
                # 원본 파일 삭제
                os.remove(source)
                self.logger.info(f"File moved successfully: {source} -> {destination}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to move file {source} to {destination}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to move file: {str(e)}",
                file_path=source,
                operation="move_file",
                original_exception=e
            )
    
    def delete_file_safely(self, file_path: str, create_backup: bool = True) -> bool:
        """
        파일 안전 삭제 (백업 생성 옵션)
        
        Args:
            file_path: 삭제할 파일 경로
            create_backup: 백업 생성 여부
            
        Returns:
            bool: 삭제 성공 여부
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"File not found for deletion: {file_path}")
                return True
            
            # 백업 생성
            if create_backup:
                backup_path = file_path + BACKUP_FILE_SUFFIX
                shutil.copy2(file_path, backup_path)
                self.logger.info(f"Backup created before deletion: {backup_path}")
            
            # 파일 삭제
            os.remove(file_path)
            self.logger.info(f"File deleted successfully: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete file {file_path}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to delete file: {str(e)}",
                file_path=file_path,
                operation="delete_file",
                original_exception=e
            )
    
    def get_file_size_mb(self, file_path: str) -> float:
        """
        파일 크기 조회 (MB 단위)
        
        Args:
            file_path: 파일 경로
            
        Returns:
            float: 파일 크기 (MB)
        """
        try:
            if not os.path.exists(file_path):
                raise FileAccessError(
                    message=f"File not found: {file_path}",
                    file_path=file_path,
                    access_type="read"
                )
            
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            return round(file_size_mb, 2)
            
        except Exception as e:
            self.logger.error(f"Failed to get file size for {file_path}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to get file size: {str(e)}",
                file_path=file_path,
                operation="get_file_size",
                original_exception=e
            )
    
    def calculate_file_hash(self, file_path: str, algorithm: str = "md5") -> str:
        """
        파일 해시 계산
        
        Args:
            file_path: 파일 경로
            algorithm: 해시 알고리즘 (md5, sha1, sha256)
            
        Returns:
            str: 파일 해시값
        """
        try:
            if not os.path.exists(file_path):
                raise FileAccessError(
                    message=f"File not found: {file_path}",
                    file_path=file_path,
                    access_type="read"
                )
            
            # 해시 알고리즘 선택
            if algorithm.lower() == "md5":
                hash_func = hashlib.md5()
            elif algorithm.lower() == "sha1":
                hash_func = hashlib.sha1()
            elif algorithm.lower() == "sha256":
                hash_func = hashlib.sha256()
            else:
                raise ValueError(f"Unsupported hash algorithm: {algorithm}")
            
            # 파일 해시 계산
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(self.buffer_size)
                    if not chunk:
                        break
                    hash_func.update(chunk)
            
            file_hash = hash_func.hexdigest()
            self.logger.debug(f"File hash calculated: {file_path} -> {file_hash}")
            
            return file_hash
            
        except Exception as e:
            self.logger.error(f"Failed to calculate file hash for {file_path}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to calculate file hash: {str(e)}",
                file_path=file_path,
                operation="calculate_hash",
                original_exception=e
            )
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        파일 메타데이터 조회
        
        Args:
            file_path: 파일 경로
            
        Returns:
            Dict[str, Any]: 파일 메타데이터
        """
        try:
            if not os.path.exists(file_path):
                raise FileAccessError(
                    message=f"File not found: {file_path}",
                    file_path=file_path,
                    access_type="read"
                )
            
            file_stat = os.stat(file_path)
            path_obj = Path(file_path)
            
            metadata = {
                'file_path': os.path.abspath(file_path),
                'file_name': path_obj.name,
                'file_stem': path_obj.stem,
                'file_extension': path_obj.suffix,
                'file_size_bytes': file_stat.st_size,
                'file_size_mb': round(file_stat.st_size / (1024 * 1024), 2),
                'created_time': datetime.fromtimestamp(file_stat.st_ctime),
                'modified_time': datetime.fromtimestamp(file_stat.st_mtime),
                'accessed_time': datetime.fromtimestamp(file_stat.st_atime),
                'permissions': stat.filemode(file_stat.st_mode),
                'owner_uid': file_stat.st_uid,
                'group_gid': file_stat.st_gid,
                'mime_type': mimetypes.guess_type(file_path)[0],
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK),
                'is_executable': os.access(file_path, os.X_OK)
            }
            
            # 해시 계산 (작은 파일만)
            if metadata['file_size_mb'] < 100:
                metadata['file_hash'] = self.calculate_file_hash(file_path)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get file metadata for {file_path}: {str(e)}")
            raise FileProcessingError(
                message=f"Failed to get file metadata: {str(e)}",
                file_path=file_path,
                operation="get_metadata",
                original_exception=e
            )
    
    def validate_file_integrity(self, file_path: str, expected_hash: Optional[str] = None) -> bool:
        """
        파일 무결성 검증
        
        Args:
            file_path: 파일 경로
            expected_hash: 예상 해시값
            
        Returns:
            bool: 무결성 검증 결과
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            # 파일 크기 검증
            file_size_mb = self.get_file_size_mb(file_path)
            if file_size_mb > MAX_FILE_SIZE_MB:
                self.logger.warning(f"File size exceeds limit: {file_size_mb}MB > {MAX_FILE_SIZE_MB}MB")
                return False
            
            # 파일 형식 검증
            file_extension = Path(file_path).suffix.lower()
            if file_extension not in self.supported_formats:
                self.logger.warning(f"Unsupported file format: {file_extension}")
                return False
            
            # 해시 검증 (제공된 경우)
            if expected_hash:
                actual_hash = self.calculate_file_hash(file_path)
                if actual_hash != expected_hash:
                    self.logger.error(f"Hash mismatch: expected {expected_hash}, got {actual_hash}")
                    return False
            
            # 파일 읽기 가능 여부 확인
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # 첫 1KB 읽기 테스트
            except Exception as e:
                self.logger.error(f"File read test failed: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"File integrity validation failed for {file_path}: {str(e)}")
            return False
    
    def _verify_file_copy(self, source: str, destination: str) -> bool:
        """
        파일 복사 검증 (내부 메서드)
        
        Args:
            source: 원본 파일 경로
            destination: 대상 파일 경로
            
        Returns:
            bool: 복사 검증 결과
        """
        try:
            # 파일 크기 비교
            source_size = os.path.getsize(source)
            dest_size = os.path.getsize(destination)
            
            if source_size != dest_size:
                return False
            
            # 해시 비교 (작은 파일만)
            if source_size < 50 * 1024 * 1024:  # 50MB 미만
                source_hash = self.calculate_file_hash(source)
                dest_hash = self.calculate_file_hash(destination)
                return source_hash == dest_hash
            
            return True
            
        except Exception as e:
            self.logger.error(f"File copy verification failed: {str(e)}")
            return False
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        파일 처리 통계 반환
        
        Returns:
            Dict[str, Any]: 처리 통계 정보
        """
        return {
            'processed_files_count': self.processed_files_count,
            'failed_files_count': self.failed_files_count,
            'total_bytes_processed': self.total_bytes_processed,
            'success_rate': (
                self.processed_files_count / 
                (self.processed_files_count + self.failed_files_count) * 100
                if (self.processed_files_count + self.failed_files_count) > 0 else 0
            )
        }
    
    def cleanup_temp_files(self, max_age_hours: int = 24) -> None:
        """
        임시 파일 정리
        
        Args:
            max_age_hours: 삭제할 파일의 최대 수명 (시간)
        """
        try:
            temp_dir = Path(self.temp_directory)
            current_time = datetime.now()
            
            for file_path in temp_dir.glob(f"*{TEMP_FILE_SUFFIX}"):
                try:
                    file_age = current_time - datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_age.total_seconds() > max_age_hours * 3600:
                        file_path.unlink()
                        self.logger.debug(f"Temp file removed: {file_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temp file {file_path}: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup temp files: {str(e)}")


# ====================================================================================
# 2. PDF 처리 클래스
# ====================================================================================

class PDFProcessor(BaseProcessor):
    """
    PDF 파일 처리 클래스
    
    PDF 파일의 읽기, 쓰기, 변환, 메타데이터 추출 등을 처리합니다.
    """
    
    def __init__(self, config: ApplicationConfig, logger):
        """
        PDFProcessor 초기화
        
        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)
        self.supported_extensions = PDF_FILE_EXTENSIONS
        
    def process(self, data: str) -> Dict[str, Any]:
        """
        PDF 파일 처리
        
        Args:
            data: PDF 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            if not self.validate_input(data):
                raise ValidationError("Invalid PDF file path")
            
            # PDF 메타데이터 추출
            metadata = self.extract_pdf_metadata(data)
            
            # 텍스트 추출
            text_content = self.extract_text_from_pdf(data)
            
            # 이미지 추출
            images = self.extract_images_from_pdf(data)
            
            result = {
                'file_path': data,
                'metadata': metadata,
                'text_content': text_content,
                'images': images,
                'processing_time': 0.0  # 실제 처리 시간 계산 필요
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"PDF processing failed for {data}: {str(e)}")
            raise FileProcessingError(
                message=f"PDF processing failed: {str(e)}",
                file_path=data,
                file_type="pdf",
                operation="process_pdf",
                original_exception=e
            )
    
    def validate_input(self, data: str) -> bool:
        """
        PDF 파일 입력 검증
        
        Args:
            data: PDF 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            if not isinstance(data, str):
                return False
            
            if not os.path.exists(data):
                return False
            
            file_extension = Path(data).suffix.lower()
            if file_extension not in self.supported_extensions:
                return False
            
            # PDF 파일 형식 검증
            return self._validate_pdf_format(data)
            
        except Exception as e:
            self.logger.error(f"PDF validation failed: {str(e)}")
            return False
    
    def extract_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        PDF 메타데이터 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Dict[str, Any]: PDF 메타데이터
        """
        try:
            # PyMuPDF 사용 (선택적 의존성)
            try:
                import fitz  # PyMuPDF
                
                pdf_document = fitz.open(pdf_path)
                metadata = pdf_document.metadata
                
                result = {
                    'title': metadata.get('title', ''),
                    'author': metadata.get('author', ''),
                    'subject': metadata.get('subject', ''),
                    'keywords': metadata.get('keywords', ''),
                    'creator': metadata.get('creator', ''),
                    'producer': metadata.get('producer', ''),
                    'creation_date': metadata.get('creationDate', ''),
                    'modification_date': metadata.get('modDate', ''),
                    'page_count': len(pdf_document),
                    'is_encrypted': pdf_document.needs_pass
                }
                
                pdf_document.close()
                return result
                
            except ImportError:
                self.logger.warning("PyMuPDF not available, using basic metadata extraction")
                return self._extract_basic_pdf_metadata(pdf_path)
            
        except Exception as e:
            self.logger.error(f"PDF metadata extraction failed: {str(e)}")
            raise FileProcessingError(
                message=f"PDF metadata extraction failed: {str(e)}",
                file_path=pdf_path,
                file_type="pdf",
                operation="extract_metadata",
                original_exception=e
            )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        PDF에서 텍스트 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            str: 추출된 텍스트
        """
        try:
            # PyMuPDF 사용
            try:
                import fitz
                
                pdf_document = fitz.open(pdf_path)
                text_content = ""
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                
                pdf_document.close()
                return text_content.strip()
                
            except ImportError:
                self.logger.warning("PyMuPDF not available for text extraction")
                return ""
            
        except Exception as e:
            self.logger.error(f"PDF text extraction failed: {str(e)}")
            raise FileProcessingError(
                message=f"PDF text extraction failed: {str(e)}",
                file_path=pdf_path,
                file_type="pdf",
                operation="extract_text",
                original_exception=e
            )
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[str]:
        """
        PDF에서 이미지 추출
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            List[str]: 추출된 이미지 파일 경로 목록
        """
        try:
            # PyMuPDF 사용
            try:
                import fitz
                
                pdf_document = fitz.open(pdf_path)
                extracted_images = []
                
                for page_num in range(len(pdf_document)):
                    page = pdf_document[page_num]
                    image_list = page.get_images(full=True)
                    
                    for img_index, img in enumerate(image_list):
                        # 이미지 추출 로직 구현
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # 이미지 저장
                        output_path = f"{pdf_path}_page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                        with open(output_path, "wb") as img_file:
                            img_file.write(image_bytes)
                        
                        extracted_images.append(output_path)
                
                pdf_document.close()
                return extracted_images
                
            except ImportError:
                self.logger.warning("PyMuPDF not available for image extraction")
                return []
            
        except Exception as e:
            self.logger.error(f"PDF image extraction failed: {str(e)}")
            raise FileProcessingError(
                message=f"PDF image extraction failed: {str(e)}",
                file_path=pdf_path,
                file_type="pdf",
                operation="extract_images",
                original_exception=e
            )
    
    def _validate_pdf_format(self, pdf_path: str) -> bool:
        """
        PDF 파일 형식 검증
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            # PDF 파일 시그니처 확인
            with open(pdf_path, 'rb') as f:
                header = f.read(4)
                if header != b'%PDF':
                    return False
            
            # PyMuPDF로 추가 검증
            try:
                import fitz
                pdf_document = fitz.open(pdf_path)
                is_valid = len(pdf_document) > 0
                pdf_document.close()
                return is_valid
            except ImportError:
                return True  # 기본 시그니처 검증만 수행
            
        except Exception as e:
            self.logger.error(f"PDF format validation failed: {str(e)}")
            return False
    
    def _extract_basic_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        기본 PDF 메타데이터 추출 (PyMuPDF 없이)
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            Dict[str, Any]: 기본 메타데이터
        """
        file_stat = os.stat(pdf_path)
        return {
            'title': Path(pdf_path).stem,
            'author': '',
            'subject': '',
            'keywords': '',
            'creator': '',
            'producer': '',
            'creation_date': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'modification_date': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'page_count': 0,
            'is_encrypted': False
        }


# ====================================================================================
# 3. 이미지 처리 클래스
# ====================================================================================

class ImageProcessor(BaseProcessor):
    """
    이미지 파일 처리 클래스
    
    이미지 파일의 변환, 크기 조정, 메타데이터 추출 등을 처리합니다.
    """
    
    def __init__(self, config: ApplicationConfig, logger):
        """
        ImageProcessor 초기화
        
        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)
        self.supported_extensions = IMAGE_FILE_EXTENSIONS
        
    def process(self, data: str) -> Dict[str, Any]:
        """
        이미지 파일 처리
        
        Args:
            data: 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            if not self.validate_input(data):
                raise ValidationError("Invalid image file path")
            
            # 이미지 메타데이터 추출
            metadata = self.extract_image_metadata(data)
            
            # 이미지 정보 분석
            image_info = self.analyze_image(data)
            
            result = {
                'file_path': data,
                'metadata': metadata,
                'image_info': image_info,
                'processing_time': 0.0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing failed for {data}: {str(e)}")
            raise FileProcessingError(
                message=f"Image processing failed: {str(e)}",
                file_path=data,
                file_type="image",
                operation="process_image",
                original_exception=e
            )
    
    def validate_input(self, data: str) -> bool:
        """
        이미지 파일 입력 검증
        
        Args:
            data: 이미지 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            if not isinstance(data, str):
                return False
            
            if not os.path.exists(data):
                return False
            
            file_extension = Path(data).suffix.lower()
            if file_extension not in self.supported_extensions:
                return False
            
            # 이미지 파일 형식 검증
            return self._validate_image_format(data)
            
        except Exception as e:
            self.logger.error(f"Image validation failed: {str(e)}")
            return False
    
    def extract_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 메타데이터 추출
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 이미지 메타데이터
        """
        try:
            # PIL 사용 (선택적 의존성)
            try:
                from PIL import Image
                from PIL.ExifTags import TAGS
                
                with Image.open(image_path) as img:
                    metadata = {
                        'format': img.format,
                        'mode': img.mode,
                        'size': img.size,
                        'width': img.width,
                        'height': img.height,
                        'has_transparency': img.mode in ('RGBA', 'LA') or 'transparency' in img.info
                    }
                    
                    # EXIF 데이터 추출
                    exif_data = {}
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = img._getexif()
                        for tag, value in exif.items():
                            decoded_tag = TAGS.get(tag, tag)
                            exif_data[decoded_tag] = value
                    
                    metadata['exif'] = exif_data
                    
                    return metadata
                    
            except ImportError:
                self.logger.warning("PIL not available for image metadata extraction")
                return self._extract_basic_image_metadata(image_path)
            
        except Exception as e:
            self.logger.error(f"Image metadata extraction failed: {str(e)}")
            raise FileProcessingError(
                message=f"Image metadata extraction failed: {str(e)}",
                file_path=image_path,
                file_type="image",
                operation="extract_metadata",
                original_exception=e
            )
    
    def analyze_image(self, image_path: str) -> Dict[str, Any]:
        """
        이미지 분석
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 이미지 분석 결과
        """
        try:
            # PIL 사용
            try:
                from PIL import Image
                import numpy as np
                
                with Image.open(image_path) as img:
                    # 기본 정보
                    info = {
                        'is_grayscale': img.mode in ('L', 'P'),
                        'is_color': img.mode in ('RGB', 'RGBA', 'CMYK'),
                        'aspect_ratio': img.width / img.height,
                        'megapixels': round((img.width * img.height) / 1_000_000, 2)
                    }
                    
                    # 이미지를 numpy 배열로 변환하여 분석
                    img_array = np.array(img)
                    
                    # 밝기 분석
                    if len(img_array.shape) == 3:
                        # 컬러 이미지
                        gray_array = np.mean(img_array, axis=2)
                    else:
                        # 그레이스케일 이미지
                        gray_array = img_array
                    
                    info['brightness'] = {
                        'mean': float(np.mean(gray_array)),
                        'std': float(np.std(gray_array)),
                        'min': float(np.min(gray_array)),
                        'max': float(np.max(gray_array))
                    }
                    
                    # 대비 분석
                    info['contrast'] = float(np.std(gray_array))
                    
                    return info
                    
            except ImportError:
                self.logger.warning("PIL/numpy not available for image analysis")
                return {'analysis_available': False}
            
        except Exception as e:
            self.logger.error(f"Image analysis failed: {str(e)}")
            return {'analysis_error': str(e)}
    
    def _validate_image_format(self, image_path: str) -> bool:
        """
        이미지 파일 형식 검증
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            # PIL로 이미지 열기 테스트
            try:
                from PIL import Image
                with Image.open(image_path) as img:
                    img.verify()
                return True
            except ImportError:
                # 기본 파일 시그니처 검증
                return self._check_image_signature(image_path)
            
        except Exception as e:
            self.logger.error(f"Image format validation failed: {str(e)}")
            return False
    
    def _check_image_signature(self, image_path: str) -> bool:
        """
        이미지 파일 시그니처 검증
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            with open(image_path, 'rb') as f:
                header = f.read(10)
                
                # 일반적인 이미지 시그니처 확인
                if header.startswith(b'\xff\xd8\xff'):  # JPEG
                    return True
                elif header.startswith(b'\x89PNG\r\n\x1a\n'):  # PNG
                    return True
                elif header.startswith(b'GIF87a') or header.startswith(b'GIF89a'):  # GIF
                    return True
                elif header.startswith(b'BM'):  # BMP
                    return True
                elif header.startswith(b'II*\x00') or header.startswith(b'MM\x00*'):  # TIFF
                    return True
                
            return False
            
        except Exception as e:
            self.logger.error(f"Image signature check failed: {str(e)}")
            return False
    
    def _extract_basic_image_metadata(self, image_path: str) -> Dict[str, Any]:
        """
        기본 이미지 메타데이터 추출 (PIL 없이)
        
        Args:
            image_path: 이미지 파일 경로
            
        Returns:
            Dict[str, Any]: 기본 메타데이터
        """
        file_stat = os.stat(image_path)
        return {
            'format': Path(image_path).suffix.upper().lstrip('.'),
            'mode': 'unknown',
            'size': (0, 0),
            'width': 0,
            'height': 0,
            'has_transparency': False,
            'exif': {},
            'file_size': file_stat.st_size,
            'created': datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat()
        }


# ====================================================================================
# 4. JSON 처리 클래스
# ====================================================================================

class JSONProcessor(BaseProcessor):
    """
    JSON 파일 처리 클래스
    
    JSON 파일의 읽기, 쓰기, 검증, 변환 등을 처리합니다.
    """
    
    def __init__(self, config: ApplicationConfig, logger):
        """
        JSONProcessor 초기화
        
        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)
        self.supported_extensions = ['.json']
        
    def process(self, data: str) -> Dict[str, Any]:
        """
        JSON 파일 처리
        
        Args:
            data: JSON 파일 경로
            
        Returns:
            Dict[str, Any]: 처리 결과
        """
        try:
            if not self.validate_input(data):
                raise ValidationError("Invalid JSON file path")
            
            # JSON 파일 읽기
            json_data = self.read_json_file(data)
            
            # JSON 구조 분석
            structure_info = self.analyze_json_structure(json_data)
            
            result = {
                'file_path': data,
                'json_data': json_data,
                'structure_info': structure_info,
                'processing_time': 0.0
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"JSON processing failed for {data}: {str(e)}")
            raise FileProcessingError(
                message=f"JSON processing failed: {str(e)}",
                file_path=data,
                file_type="json",
                operation="process_json",
                original_exception=e
            )
    
    def validate_input(self, data: str) -> bool:
        """
        JSON 파일 입력 검증
        
        Args:
            data: JSON 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            if not isinstance(data, str):
                return False
            
            if not os.path.exists(data):
                return False
            
            file_extension = Path(data).suffix.lower()
            if file_extension not in self.supported_extensions:
                return False
            
            # JSON 파일 형식 검증
            return self._validate_json_format(data)
            
        except Exception as e:
            self.logger.error(f"JSON validation failed: {str(e)}")
            return False
    
    def read_json_file(self, json_path: str) -> Any:
        """
        JSON 파일 읽기
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            Any: JSON 데이터
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except json.JSONDecodeError as e:
            self.logger.error(f"JSON decode error: {str(e)}")
            raise FileProcessingError(
                message=f"Invalid JSON format: {str(e)}",
                file_path=json_path,
                file_type="json",
                operation="read_json",
                original_exception=e
            )
        except Exception as e:
            self.logger.error(f"JSON read failed: {str(e)}")
            raise FileProcessingError(
                message=f"JSON read failed: {str(e)}",
                file_path=json_path,
                file_type="json",
                operation="read_json",
                original_exception=e
            )
    
    def write_json_file(self, data: Any, json_path: str, indent: int = 2) -> bool:
        """
        JSON 파일 쓰기
        
        Args:
            data: 저장할 데이터
            json_path: JSON 파일 경로
            indent: 들여쓰기 수준
            
        Returns:
            bool: 쓰기 성공 여부
        """
        try:
            # 디렉터리 생성
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            
            # JSON 파일 쓰기
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=indent)
            
            return True
            
        except Exception as e:
            self.logger.error(f"JSON write failed: {str(e)}")
            raise FileProcessingError(
                message=f"JSON write failed: {str(e)}",
                file_path=json_path,
                file_type="json",
                operation="write_json",
                original_exception=e
            )
    
    def analyze_json_structure(self, json_data: Any) -> Dict[str, Any]:
        """
        JSON 구조 분석
        
        Args:
            json_data: JSON 데이터
            
        Returns:
            Dict[str, Any]: 구조 분석 결과
        """
        try:
            def analyze_value(value: Any, path: str = "") -> Dict[str, Any]:
                if isinstance(value, dict):
                    return {
                        'type': 'object',
                        'keys': list(value.keys()),
                        'key_count': len(value),
                        'nested_objects': sum(1 for v in value.values() if isinstance(v, dict)),
                        'nested_arrays': sum(1 for v in value.values() if isinstance(v, list))
                    }
                elif isinstance(value, list):
                    return {
                        'type': 'array',
                        'length': len(value),
                        'item_types': list(set(type(item).__name__ for item in value)),
                        'uniform_type': len(set(type(item).__name__ for item in value)) == 1
                    }
                else:
                    return {
                        'type': type(value).__name__,
                        'value': value if not isinstance(value, (str, int, float, bool)) else str(value)[:100]
                    }
            
            return analyze_value(json_data)
            
        except Exception as e:
            self.logger.error(f"JSON structure analysis failed: {str(e)}")
            return {'analysis_error': str(e)}
    
    def _validate_json_format(self, json_path: str) -> bool:
        """
        JSON 파일 형식 검증
        
        Args:
            json_path: JSON 파일 경로
            
        Returns:
            bool: 검증 결과
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
            
        except json.JSONDecodeError:
            return False
        except Exception as e:
            self.logger.error(f"JSON format validation failed: {str(e)}")
            return False


# ====================================================================================
# 5. 압축 처리 클래스
# ====================================================================================

class CompressionHandler:
    """
    파일 압축 및 해제 처리 클래스
    
    ZIP, TAR, GZIP 등의 압축 형식을 지원합니다.
    """
    
    def __init__(self, config: Optional[ApplicationConfig] = None):
        """
        CompressionHandler 초기화
        
        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger('compression_handler')
        self.supported_formats = ['.zip', '.tar', '.tar.gz', '.tar.bz2', '.gz']
        
    def compress_file(self, file_path: str, compression_level: int = 6) -> str:
        """
        파일 압축
        
        Args:
            file_path: 압축할 파일 경로
            compression_level: 압축 레벨 (1-9)
            
        Returns:
            str: 압축 파일 경로
        """
        try:
            if not os.path.exists(file_path):
                raise FileAccessError(
                    message=f"File not found: {file_path}",
                    file_path=file_path,
                    access_type="read"
                )
            
            # 압축 파일 경로 생성
            compressed_path = file_path + '.gz'
            
            # GZIP 압축 수행
            with open(file_path, 'rb') as f_in:
                with gzip.open(compressed_path, 'wb', compresslevel=compression_level) as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # 압축 결과 검증
            if not os.path.exists(compressed_path):
                raise FileProcessingError(
                    message="Compression failed - output file not created",
                    file_path=file_path,
                    operation="compress_file"
                )
            
            self.logger.info(f"File compressed: {file_path} -> {compressed_path}")
            return compressed_path
            
        except Exception as e:
            self.logger.error(f"File compression failed: {str(e)}")
            raise FileProcessingError(
                message=f"File compression failed: {str(e)}",
                file_path=file_path,
                operation="compress_file",
                original_exception=e
            )
    
    def extract_compressed_file(self, compressed_path: str, extract_to: str) -> bool:
        """
        압축 파일 해제
        
        Args:
            compressed_path: 압축 파일 경로
            extract_to: 압축 해제 디렉터리
            
        Returns:
            bool: 해제 성공 여부
        """
        try:
            if not os.path.exists(compressed_path):
                raise FileAccessError(
                    message=f"Compressed file not found: {compressed_path}",
                    file_path=compressed_path,
                    access_type="read"
                )
            
            # 압축 해제 디렉터리 생성
            os.makedirs(extract_to, exist_ok=True)
            
            # 압축 형식에 따른 해제
            if compressed_path.endswith('.zip'):
                return self._extract_zip(compressed_path, extract_to)
            elif compressed_path.endswith('.tar') or compressed_path.endswith('.tar.gz') or compressed_path.endswith('.tar.bz2'):
                return self._extract_tar(compressed_path, extract_to)
            elif compressed_path.endswith('.gz'):
                return self._extract_gzip(compressed_path, extract_to)
            else:
                raise FileProcessingError(
                    message=f"Unsupported compression format: {compressed_path}",
                    file_path=compressed_path,
                    operation="extract_compressed_file"
                )
            
        except Exception as e:
            self.logger.error(f"File extraction failed: {str(e)}")
            raise FileProcessingError(
                message=f"File extraction failed: {str(e)}",
                file_path=compressed_path,
                operation="extract_compressed_file",
                original_exception=e
            )
    
    def _extract_zip(self, zip_path: str, extract_to: str) -> bool:
        """ZIP 파일 해제"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            self.logger.error(f"ZIP extraction failed: {str(e)}")
            return False
    
    def _extract_tar(self, tar_path: str, extract_to: str) -> bool:
        """TAR 파일 해제"""
        try:
            with tarfile.open(tar_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)
            return True
        except Exception as e:
            self.logger.error(f"TAR extraction failed: {str(e)}")
            return False
    
    def _extract_gzip(self, gz_path: str, extract_to: str) -> bool:
        """GZIP 파일 해제"""
        try:
            output_path = os.path.join(extract_to, os.path.basename(gz_path).replace('.gz', ''))
            
            with gzip.open(gz_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            return True
        except Exception as e:
            self.logger.error(f"GZIP extraction failed: {str(e)}")
            return False


# ====================================================================================
# 6. 유틸리티 함수들
# ====================================================================================

def create_directory_if_not_exists(directory_path: str) -> bool:
    """
    디렉터리가 존재하지 않으면 생성
    
    Args:
        directory_path: 생성할 디렉터리 경로
        
    Returns:
        bool: 생성 성공 여부
    """
    handler = FileHandler()
    return handler.create_directory_if_not_exists(directory_path)


def copy_file_with_backup(source: str, destination: str) -> bool:
    """
    파일 복사 (기존 파일이 있으면 백업)
    
    Args:
        source: 원본 파일 경로
        destination: 대상 파일 경로
        
    Returns:
        bool: 복사 성공 여부
    """
    handler = FileHandler()
    return handler.copy_file_with_backup(source, destination)


def get_file_size_mb(file_path: str) -> float:
    """
    파일 크기 조회 (MB 단위)
    
    Args:
        file_path: 파일 경로
        
    Returns:
        float: 파일 크기 (MB)
    """
    handler = FileHandler()
    return handler.get_file_size_mb(file_path)


def calculate_file_hash(file_path: str) -> str:
    """
    파일 해시 계산
    
    Args:
        file_path: 파일 경로
        
    Returns:
        str: 파일 해시값
    """
    handler = FileHandler()
    return handler.calculate_file_hash(file_path)


def compress_file(file_path: str, compression_level: int = 6) -> str:
    """
    파일 압축
    
    Args:
        file_path: 압축할 파일 경로
        compression_level: 압축 레벨
        
    Returns:
        str: 압축 파일 경로
    """
    compressor = CompressionHandler()
    return compressor.compress_file(file_path, compression_level)


def extract_compressed_file(compressed_path: str, extract_to: str) -> bool:
    """
    압축 파일 해제
    
    Args:
        compressed_path: 압축 파일 경로
        extract_to: 압축 해제 디렉터리
        
    Returns:
        bool: 해제 성공 여부
    """
    compressor = CompressionHandler()
    return compressor.extract_compressed_file(compressed_path, extract_to)


def get_file_metadata(file_path: str) -> Dict[str, Any]:
    """
    파일 메타데이터 조회
    
    Args:
        file_path: 파일 경로
        
    Returns:
        Dict[str, Any]: 파일 메타데이터
    """
    handler = FileHandler()
    return handler.get_file_metadata(file_path)


def validate_file_integrity(file_path: str, expected_hash: Optional[str] = None) -> bool:
    """
    파일 무결성 검증
    
    Args:
        file_path: 파일 경로
        expected_hash: 예상 해시값
        
    Returns:
        bool: 무결성 검증 결과
    """
    handler = FileHandler()
    return handler.validate_file_integrity(file_path, expected_hash)


def process_pdf_file(pdf_path: str, config: ApplicationConfig) -> Dict[str, Any]:
    """
    PDF 파일 처리
    
    Args:
        pdf_path: PDF 파일 경로
        config: 설정 객체
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    logger = get_application_logger('pdf_processor')
    processor = PDFProcessor(config, logger)
    return processor.process(pdf_path)


def process_image_file(image_path: str, config: ApplicationConfig) -> Dict[str, Any]:
    """
    이미지 파일 처리
    
    Args:
        image_path: 이미지 파일 경로
        config: 설정 객체
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    logger = get_application_logger('image_processor')
    processor = ImageProcessor(config, logger)
    return processor.process(image_path)


def process_json_file(json_path: str, config: ApplicationConfig) -> Dict[str, Any]:
    """
    JSON 파일 처리
    
    Args:
        json_path: JSON 파일 경로
        config: 설정 객체
        
    Returns:
        Dict[str, Any]: 처리 결과
    """
    logger = get_application_logger('json_processor')
    processor = JSONProcessor(config, logger)
    return processor.process(json_path)


# ====================================================================================
# 7. 런타임 검증 및 테스트
# ====================================================================================

def validate_file_handlers() -> bool:
    """
    파일 핸들러 유효성 검증
    
    Returns:
        bool: 검증 성공 여부
    """
    try:
        # 기본 파일 핸들러 테스트
        handler = FileHandler()
        
        # 임시 파일 생성 및 테스트
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write("test content")
            tmp_path = tmp_file.name
        
        try:
            # 메타데이터 조회 테스트
            metadata = handler.get_file_metadata(tmp_path)
            if not metadata:
                return False
            
            # 파일 크기 조회 테스트
            size = handler.get_file_size_mb(tmp_path)
            if size <= 0:
                return False
            
            # 해시 계산 테스트
            file_hash = handler.calculate_file_hash(tmp_path)
            if not file_hash:
                return False
            
            # 무결성 검증 테스트
            if not handler.validate_file_integrity(tmp_path, file_hash):
                return False
            
            return True
            
        finally:
            # 임시 파일 삭제
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except Exception as e:
        print(f"File handler validation failed: {e}")
        return False


if __name__ == "__main__":
    # 파일 핸들러 테스트
    print("YOKOGAWA OCR 파일 핸들러 테스트")
    print("=" * 50)
    
    try:
        # 파일 핸들러 검증
        if validate_file_handlers():
            print("✅ 파일 핸들러 검증 완료")
        else:
            print("❌ 파일 핸들러 검증 실패")
        
        # 기본 기능 테스트
        handler = FileHandler()
        
        # 임시 디렉터리 생성 테스트
        temp_dir = tempfile.mkdtemp()
        test_dir = os.path.join(temp_dir, "test_directory")
        
        if handler.create_directory_if_not_exists(test_dir):
            print("✅ 디렉터리 생성 테스트 완료")
        
        # 임시 파일 생성 및 복사 테스트
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir) as tmp_file:
            tmp_file.write("Hello, YOKOGAWA OCR!")
            source_path = tmp_file.name
        
        destination_path = os.path.join(test_dir, "copied_file.txt")
        
        if handler.copy_file_with_backup(source_path, destination_path):
            print("✅ 파일 복사 테스트 완료")
        
        # 파일 메타데이터 조회 테스트
        metadata = handler.get_file_metadata(destination_path)
        print(f"✅ 파일 메타데이터 조회 완료: {metadata['file_name']}")
        
        # 압축 테스트
        compressor = CompressionHandler()
        compressed_path = compressor.compress_file(destination_path)
        print(f"✅ 파일 압축 테스트 완료: {compressed_path}")
        
        # 처리 통계 출력
        stats = handler.get_processing_statistics()
        print(f"📊 처리 통계: {stats}")
        
        # 임시 파일 정리
        shutil.rmtree(temp_dir)
        
    except Exception as e:
        print(f"❌ 파일 핸들러 테스트 실패: {e}")
    
    print("\n🎯 파일 핸들러 구현이 완료되었습니다!")
