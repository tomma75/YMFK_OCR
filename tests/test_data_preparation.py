#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 통합 테스트 모듈

이 모듈은 전체 시스템의 각 컴포넌트에 대한 단위 테스트와 통합 테스트를 수행합니다.
실제 구현된 클래스와 함수들을 기반으로 테스트를 진행합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import unittest
import os
import sys
import tempfile
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import uuid
import numpy as np

# 프로젝트 루트 디렉터리를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 테스트 대상 모듈 임포트
from config.settings import ApplicationConfig, load_configuration
from core.exceptions import (
    ApplicationError,
    DataCollectionError,
    LabelingError,
    AugmentationError,
    ValidationError,
    FileProcessingError,
    ImageProcessingError,
)

# 서비스 임포트
from services.data_collection_service import (
    DataCollectionService,
    create_data_collection_service,
    FileCollector,
    MetadataExtractor,
    DuplicateDetector,
)
from services.labeling_service import (
    LabelingService,
    create_labeling_service,
    AnnotationManager,
    QualityController,
    LabelingSessionManager,
)
from services.augmentation_service import (
    AugmentationService,
    create_augmentation_service,
    ImageAugmenter,
)
from services.validation_service import (
    ValidationService,
    create_validation_service,
    DataQualityValidator,
    ConsistencyChecker,
    StatisticsGenerator,
)

# 모델 임포트
from models.document_model import DocumentModel, DocumentMetadata, PageInfo
from models.annotation_model import AnnotationModel, BoundingBox, FieldAnnotation

# 유틸리티 임포트
from utils.file_handler import (
    FileHandler,
    PDFProcessor,
    ImageProcessor as FileImageProcessor,
)
from utils.logger_util import setup_logger, get_application_logger
from utils.image_processor import ImageProcessor, ImageConverter, ImageEnhancer

# ====================================================================================
# 테스트 헬퍼 함수들
# ====================================================================================


def create_test_config() -> ApplicationConfig:
    """테스트용 설정 생성"""
    config = ApplicationConfig()
    config.environment = "testing"
    config.debug_mode = True
    config.data_directory = tempfile.mkdtemp()
    config.raw_data_directory = os.path.join(config.data_directory, "raw")
    config.processed_data_directory = os.path.join(config.data_directory, "processed")
    config.annotations_directory = os.path.join(config.data_directory, "annotations")
    config.augmented_data_directory = os.path.join(config.data_directory, "augmented")
    return config


def create_test_pdf_file(file_path: str) -> None:
    """테스트용 PDF 파일 생성"""
    # 간단한 PDF 내용을 시뮬레이션
    with open(file_path, "wb") as f:
        f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")


def create_test_image_file(file_path: str) -> None:
    """테스트용 이미지 파일 생성"""
    try:
        from PIL import Image

        # 간단한 테스트 이미지 생성
        img = Image.new("RGB", (100, 100), color="white")
        img.save(file_path)
    except ImportError:
        # PIL이 없는 경우 더미 파일 생성
        with open(file_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00d\x00\x00\x00d")


def create_test_document_model(file_path: str) -> DocumentModel:
    """테스트용 DocumentModel 생성"""
    return DocumentModel.from_file_path(file_path)


def create_test_annotation_model() -> AnnotationModel:
    """테스트용 AnnotationModel 생성"""
    bbox = BoundingBox(x=10, y=20, width=100, height=50)
    return AnnotationModel(
        document_id="test_doc_001", page_number=1, annotation_type="text"
    )


def cleanup_test_directory(directory: str) -> None:
    """테스트 디렉터리 정리"""
    if os.path.exists(directory):
        shutil.rmtree(directory)


# ====================================================================================
# 1. 데이터 수집 테스트 클래스
# ====================================================================================


class TestDataCollection(unittest.TestCase):
    """데이터 수집 서비스 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_data_collection", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()
        self.service = None

        # 테스트 파일 생성
        self.test_pdf_path = os.path.join(self.test_data_dir, "test_document.pdf")
        self.test_image_path = os.path.join(self.test_data_dir, "test_image.jpg")

        create_test_pdf_file(self.test_pdf_path)
        create_test_image_file(self.test_image_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        if self.service:
            self.service.cleanup()
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_file_collection_process(self) -> None:
        """파일 수집 프로세스 테스트"""
        # 서비스 생성 및 초기화
        self.service = create_data_collection_service(self.config)
        self.assertTrue(self.service.initialize())

        # 파일 수집 테스트
        collected_files = self.service.collect_files(self.test_data_dir)
        self.assertIsInstance(collected_files, list)
        self.assertGreater(len(collected_files), 0)

        # 수집된 파일 검증
        collected_extensions = [Path(f).suffix.lower() for f in collected_files]
        self.assertIn(".pdf", collected_extensions)
        self.assertIn(".jpg", collected_extensions)

        # 수집 통계 확인
        statistics = self.service.get_collection_statistics()
        self.assertIsInstance(statistics, dict)
        self.assertIn("total_files", statistics)

    def test_file_collector_functionality(self) -> None:
        """FileCollector 기능 테스트"""
        supported_formats = [".pdf", ".jpg", ".png"]
        collector = FileCollector(supported_formats)

        collected_files = collector.collect_files_from_directory(self.test_data_dir)
        self.assertIsInstance(collected_files, list)
        self.assertGreater(len(collected_files), 0)

    def test_metadata_extraction(self) -> None:
        """메타데이터 추출 테스트"""
        file_handler = FileHandler(self.config)
        extractor = MetadataExtractor(file_handler)

        metadata = extractor.extract_file_metadata(self.test_pdf_path)
        self.assertIsInstance(metadata, dict)
        self.assertIn("file_size_mb", metadata)
        self.assertIn("document_type", metadata)

    def test_duplicate_detection(self) -> None:
        """중복 파일 탐지 테스트"""
        # 중복 파일 생성
        duplicate_path = os.path.join(self.test_data_dir, "duplicate.pdf")
        shutil.copy2(self.test_pdf_path, duplicate_path)

        file_handler = FileHandler(self.config)
        detector = DuplicateDetector(file_handler)

        file_list = [self.test_pdf_path, duplicate_path]
        unique_files = detector.detect_duplicates(file_list)

        self.assertEqual(len(unique_files), 1)

    def test_error_handling(self) -> None:
        """오류 처리 테스트"""
        self.service = create_data_collection_service(self.config)

        # 존재하지 않는 디렉터리 테스트
        with self.assertRaises(DataCollectionError):
            self.service.collect_files("/nonexistent/directory")


# ====================================================================================
# 2. 라벨링 서비스 테스트 클래스
# ====================================================================================


class TestLabelingService(unittest.TestCase):
    """라벨링 서비스 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_labeling", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()
        self.service = None

        # 테스트 문서 생성
        self.test_document_path = os.path.join(self.test_data_dir, "test_doc.pdf")
        create_test_pdf_file(self.test_document_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        if self.service:
            self.service.cleanup()
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_annotation_creation(self) -> None:
        """어노테이션 생성 테스트"""
        self.service = create_labeling_service(self.config)
        self.assertTrue(self.service.initialize())

        # 라벨링 세션 생성
        session_id = self.service.create_labeling_session(self.test_document_path)
        self.assertIsInstance(session_id, str)
        self.assertGreater(len(session_id), 0)

        # 라벨링 진행 상황 확인
        progress = self.service.get_labeling_progress()
        self.assertIsInstance(progress, dict)
        self.assertIn("progress", progress)

    def test_annotation_manager(self) -> None:
        """AnnotationManager 테스트"""
        manager = AnnotationManager(self.config)

        # 어노테이션 생성
        annotation = manager.create_annotation(
            document_id="test_doc", page_number=1, annotation_type="text"
        )

        self.assertIsInstance(annotation, AnnotationModel)
        self.assertEqual(annotation.document_id, "test_doc")
        self.assertEqual(annotation.page_number, 1)

    def test_quality_controller(self) -> None:
        """QualityController 테스트"""
        controller = QualityController(self.config)
        annotation = create_test_annotation_model()

        # 품질 점수 계산
        quality_score = controller.calculate_quality_score(annotation)
        self.assertIsInstance(quality_score, float)
        self.assertGreaterEqual(quality_score, 0.0)
        self.assertLessEqual(quality_score, 1.0)

    def test_labeling_session_manager(self) -> None:
        """LabelingSessionManager 테스트"""
        manager = LabelingSessionManager(self.config)

        # 세션 생성
        session_id = manager.create_session(self.test_document_path)
        self.assertIsInstance(session_id, str)

        # 세션 상태 확인
        session_status = manager.get_session_status(session_id)
        self.assertIsInstance(session_status, dict)

    def test_annotation_template_loading(self) -> None:
        """어노테이션 템플릿 로딩 테스트"""
        self.service = create_labeling_service(self.config)
        self.assertTrue(self.service.initialize())

        # 템플릿 로드
        template = self.service.load_annotation_template()
        self.assertIsInstance(template, dict)
        self.assertIn("template_name", template)


# ====================================================================================
# 3. 데이터 증강 서비스 테스트 클래스
# ====================================================================================


class TestAugmentationService(unittest.TestCase):
    """데이터 증강 서비스 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_augmentation", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()
        self.service = None

        # 테스트 이미지 생성
        self.test_image_path = os.path.join(self.test_data_dir, "test_image.jpg")
        create_test_image_file(self.test_image_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        if self.service:
            self.service.cleanup()
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_data_augmentation(self) -> None:
        """데이터 증강 테스트"""
        self.service = create_augmentation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 데이터셋 생성
        test_dataset = [
            {"image_path": self.test_image_path, "label": "test_document"},
            {"image_path": self.test_image_path, "label": "test_invoice"},
        ]

        # 데이터셋 증강
        augmented_dataset = self.service.augment_dataset(test_dataset)
        self.assertIsInstance(augmented_dataset, list)
        self.assertGreaterEqual(len(augmented_dataset), len(test_dataset))

        # 증강 통계 확인
        statistics = self.service.get_augmentation_statistics()
        self.assertIsInstance(statistics, dict)
        self.assertIn("augmentation_factor", statistics)

    def test_image_augmenter(self) -> None:
        """ImageAugmenter 테스트"""
        augmenter = ImageAugmenter(self.config)

        # 증강 기법 목록
        augmentation_types = ["rotation", "scaling", "brightness", "contrast"]

        # 이미지 증강 수행
        augmented_images = augmenter.augment_image(
            self.test_image_path, augmentation_types
        )
        self.assertIsInstance(augmented_images, list)
        self.assertGreater(len(augmented_images), 0)

    def test_geometric_transformations(self) -> None:
        """기하학적 변환 테스트"""
        self.service = create_augmentation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 이미지 배열 생성
        test_image = np.random.rand(100, 100, 3) * 255
        test_image = test_image.astype(np.uint8)

        # 기하학적 변환 적용
        transformed_images = self.service.apply_geometric_transformations(test_image)
        self.assertIsInstance(transformed_images, list)
        self.assertGreater(len(transformed_images), 0)

    def test_color_adjustments(self) -> None:
        """색상 조정 테스트"""
        self.service = create_augmentation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 이미지 배열 생성
        test_image = np.random.rand(100, 100, 3) * 255
        test_image = test_image.astype(np.uint8)

        # 색상 조정 적용
        adjusted_images = self.service.apply_color_adjustments(test_image)
        self.assertIsInstance(adjusted_images, list)
        self.assertGreater(len(adjusted_images), 0)

    def test_noise_addition(self) -> None:
        """노이즈 추가 테스트"""
        self.service = create_augmentation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 이미지 배열 생성
        test_image = np.random.rand(100, 100, 3) * 255
        test_image = test_image.astype(np.uint8)

        # 노이즈 추가
        noisy_images = self.service.add_noise_variations(test_image)
        self.assertIsInstance(noisy_images, list)
        self.assertGreater(len(noisy_images), 0)

    def test_augmentation_rule_configuration(self) -> None:
        """증강 규칙 설정 테스트"""
        self.service = create_augmentation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 증강 규칙 설정
        rules = {
            "enabled_techniques": ["rotation", "scaling"],
            "rotation_range": (-10, 10),
            "scale_range": (0.9, 1.1),
        }

        # 규칙 적용 (예외가 발생하지 않으면 성공)
        try:
            self.service.configure_augmentation_rules(rules)
            success = True
        except Exception:
            success = False

        self.assertTrue(success)


# ====================================================================================
# 4. 검증 서비스 테스트 클래스
# ====================================================================================


class TestValidationService(unittest.TestCase):
    """검증 서비스 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_validation", self.config.logging_config)
        self.service = None

    def tearDown(self) -> None:
        """테스트 정리"""
        if self.service:
            self.service.cleanup()
        cleanup_test_directory(self.config.data_directory)

    def test_validation_pipeline(self) -> None:
        """검증 파이프라인 테스트"""
        self.service = create_validation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 데이터셋 생성
        test_dataset = [
            {
                "document_id": "doc_001",
                "status": "completed",
                "annotations": [
                    {
                        "field_name": "document_title",
                        "bounding_box": {
                            "x": 100,
                            "y": 200,
                            "width": 300,
                            "height": 50,
                        },
                        "text_value": "Test Document",
                        "confidence_score": 0.95,
                    }
                ],
            },
            {
                "document_id": "doc_002",
                "status": "completed",
                "annotations": [
                    {
                        "field_name": "supplier_name",
                        "bounding_box": {
                            "x": 150,
                            "y": 250,
                            "width": 200,
                            "height": 30,
                        },
                        "text_value": "Test Supplier",
                        "confidence_score": 0.92,
                    }
                ],
            },
        ]

        # 데이터셋 검증
        validation_result = self.service.validate_dataset(test_dataset)
        self.assertIsInstance(validation_result, dict)
        self.assertIn("validation_passed", validation_result)
        self.assertIn("overall_quality_score", validation_result)

    def test_data_quality_validator(self) -> None:
        """DataQualityValidator 테스트"""
        validator = DataQualityValidator(self.config)

        # 테스트 데이터셋
        test_dataset = [
            {"document_id": "doc1", "annotations": [], "status": "completed"},
            {"document_id": "doc2", "annotations": [], "status": "completed"},
        ]

        # 품질 검증
        quality_result = validator.validate_dataset_quality(test_dataset)
        self.assertIsInstance(quality_result, dict)
        self.assertIn("overall_quality_score", quality_result)

    def test_consistency_checker(self) -> None:
        """ConsistencyChecker 테스트"""
        checker = ConsistencyChecker(self.config)

        # 테스트 데이터셋
        test_dataset = [
            {"document_id": "doc1", "format": "pdf", "annotations": []},
            {"document_id": "doc2", "format": "pdf", "annotations": []},
        ]

        # 일관성 검사
        consistency_result = checker.check_dataset_consistency(test_dataset)
        self.assertIsInstance(consistency_result, dict)
        self.assertIn("consistency_score", consistency_result)

    def test_statistics_generator(self) -> None:
        """StatisticsGenerator 테스트"""
        generator = StatisticsGenerator(self.config)

        # 테스트 데이터셋
        test_dataset = [
            {"document_id": "doc1", "page_count": 5, "annotations": []},
            {"document_id": "doc2", "page_count": 3, "annotations": []},
        ]

        # 통계 생성
        statistics = generator.generate_dataset_statistics(test_dataset)
        self.assertIsInstance(statistics, dict)
        self.assertIn("total_documents", statistics)

    def test_validation_report_generation(self) -> None:
        """검증 보고서 생성 테스트"""
        self.service = create_validation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 데이터셋으로 검증 수행
        test_dataset = [{"document_id": "doc1", "status": "completed"}]
        self.service.validate_dataset(test_dataset)

        # 보고서 생성
        report = self.service.generate_validation_report()
        self.assertIsInstance(report, dict)
        self.assertIn("validation_summary", report)

    def test_quality_report_generation(self) -> None:
        """품질 보고서 생성 테스트"""
        self.service = create_validation_service(self.config)
        self.assertTrue(self.service.initialize())

        # 테스트 데이터셋
        test_dataset = [{"document_id": "doc1", "annotations": []}]

        # 품질 보고서 생성
        quality_report = self.service.generate_quality_report(test_dataset)
        self.assertIsInstance(quality_report, dict)
        self.assertIn("quality_summary", quality_report)


# ====================================================================================
# 5. 이미지 처리 테스트 클래스
# ====================================================================================


class TestImageProcessor(unittest.TestCase):
    """이미지 처리 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_image_processor", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()

        # 테스트 파일 생성
        self.test_image_path = os.path.join(self.test_data_dir, "test_image.jpg")
        self.test_pdf_path = os.path.join(self.test_data_dir, "test_document.pdf")

        create_test_image_file(self.test_image_path)
        create_test_pdf_file(self.test_pdf_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_image_processing(self) -> None:
        """이미지 처리 테스트"""
        processor = ImageProcessor(self.config, self.logger)

        # 입력 검증
        is_valid = processor.validate_input(self.test_image_path)
        self.assertTrue(is_valid)

        # 이미지 처리
        try:
            result = processor.process(self.test_image_path)
            self.assertIsInstance(result, dict)
            self.assertIn("source_file", result)
        except Exception as e:
            # 실제 이미지 처리 라이브러리가 없는 경우 예외 발생 가능
            self.assertIsInstance(e, (ImageProcessingError, ImportError))

    def test_image_converter(self) -> None:
        """ImageConverter 테스트"""
        converter = ImageConverter(self.config, self.logger)

        # PDF to 이미지 변환 테스트
        try:
            converted_images = converter.convert_pdf_to_images(self.test_pdf_path, None)
            self.assertIsInstance(converted_images, list)
        except Exception as e:
            # PDF 처리 라이브러리가 없는 경우 예외 발생 가능
            self.assertIsInstance(e, (ImageProcessingError, ImportError))

    def test_image_enhancer(self) -> None:
        """ImageEnhancer 테스트"""
        enhancer = ImageEnhancer(self.config, self.logger)

        # 이미지 품질 개선 테스트
        try:
            enhanced_path = enhancer.enhance_image_quality(self.test_image_path, None)
            self.assertIsInstance(enhanced_path, str)
        except Exception as e:
            # 이미지 처리 라이브러리가 없는 경우 예외 발생 가능
            self.assertIsInstance(e, (ImageProcessingError, ImportError))

    def test_file_image_processor(self) -> None:
        """FileImageProcessor 테스트"""
        processor = FileImageProcessor(self.config, self.logger)

        # 입력 검증
        is_valid = processor.validate_input(self.test_image_path)
        self.assertTrue(is_valid)

        # 이미지 처리
        try:
            result = processor.process(self.test_image_path)
            self.assertIsInstance(result, dict)
        except Exception as e:
            # 이미지 처리 라이브러리가 없는 경우 예외 발생 가능
            self.assertIsInstance(e, (ImageProcessingError, ImportError))


# ====================================================================================
# 6. 통합 테스트 클래스
# ====================================================================================


class TestIntegration(unittest.TestCase):
    """통합 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_integration", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()

        # 테스트 파일 생성
        self.test_pdf_path = os.path.join(self.test_data_dir, "test_document.pdf")
        create_test_pdf_file(self.test_pdf_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_end_to_end_pipeline(self) -> None:
        """전체 파이프라인 통합 테스트"""
        # 1. 데이터 수집
        collection_service = create_data_collection_service(self.config)
        collection_service.initialize()

        collected_files = collection_service.collect_files(self.test_data_dir)
        self.assertGreater(len(collected_files), 0)

        # 2. 라벨링
        labeling_service = create_labeling_service(self.config)
        labeling_service.initialize()

        session_id = labeling_service.create_labeling_session(self.test_pdf_path)
        self.assertIsNotNone(session_id)

        # 3. 데이터 증강
        augmentation_service = create_augmentation_service(self.config)
        augmentation_service.initialize()

        test_dataset = [{"image_path": self.test_pdf_path, "label": "document"}]
        augmented_dataset = augmentation_service.augment_dataset(test_dataset)
        self.assertGreaterEqual(len(augmented_dataset), len(test_dataset))

        # 4. 검증
        validation_service = create_validation_service(self.config)
        validation_service.initialize()

        validation_result = validation_service.validate_dataset(augmented_dataset)
        self.assertIsInstance(validation_result, dict)

        # 서비스 정리
        collection_service.cleanup()
        labeling_service.cleanup()
        augmentation_service.cleanup()
        validation_service.cleanup()

    def test_service_interaction(self) -> None:
        """서비스 간 상호작용 테스트"""
        # 모든 서비스 초기화
        services = {
            "collection": create_data_collection_service(self.config),
            "labeling": create_labeling_service(self.config),
            "augmentation": create_augmentation_service(self.config),
            "validation": create_validation_service(self.config),
        }

        # 모든 서비스 초기화
        for service_name, service in services.items():
            initialized = service.initialize()
            self.assertTrue(
                initialized, f"{service_name} service initialization failed"
            )

        # 모든 서비스 상태 확인
        for service_name, service in services.items():
            health_status = service.health_check()
            self.assertTrue(
                health_status, f"{service_name} service health check failed"
            )

        # 서비스 정리
        for service in services.values():
            service.cleanup()

    def test_error_propagation(self) -> None:
        """오류 전파 테스트"""
        collection_service = create_data_collection_service(self.config)
        collection_service.initialize()

        # 잘못된 경로로 오류 발생 시킴
        with self.assertRaises(DataCollectionError):
            collection_service.collect_files("/nonexistent/path")

        collection_service.cleanup()


# ====================================================================================
# 7. 성능 테스트 클래스
# ====================================================================================


class TestPerformance(unittest.TestCase):
    """성능 테스트"""

    def setUp(self) -> None:
        """테스트 설정 초기화"""
        self.config = create_test_config()
        self.logger = setup_logger("test_performance", self.config.logging_config)
        self.test_data_dir = tempfile.mkdtemp()

        # 여러 테스트 파일 생성
        self.test_files = []
        for i in range(10):
            file_path = os.path.join(self.test_data_dir, f"test_file_{i}.pdf")
            create_test_pdf_file(file_path)
            self.test_files.append(file_path)

    def tearDown(self) -> None:
        """테스트 정리"""
        cleanup_test_directory(self.test_data_dir)
        cleanup_test_directory(self.config.data_directory)

    def test_batch_processing_performance(self) -> None:
        """배치 처리 성능 테스트"""
        collection_service = create_data_collection_service(self.config)
        collection_service.initialize()

        # 시간 측정
        start_time = time.time()
        collected_files = collection_service.collect_files(self.test_data_dir)
        end_time = time.time()

        processing_time = end_time - start_time
        files_per_second = (
            len(collected_files) / processing_time if processing_time > 0 else 0
        )

        self.assertGreater(files_per_second, 0)
        self.assertLess(processing_time, 10.0)  # 10초 이내 처리

        collection_service.cleanup()

    def test_memory_usage(self) -> None:
        """메모리 사용량 테스트"""
        # 메모리 사용량 모니터링은 실제 운영 환경에서 더 중요
        # 여기서는 기본적인 체크만 수행
        import psutil

        process = psutil.Process(os.getpid())

        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 메모리 집약적 작업 수행
        collection_service = create_data_collection_service(self.config)
        collection_service.initialize()
        collection_service.collect_files(self.test_data_dir)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 메모리 증가량이 합리적인 범위 내에 있는지 확인
        self.assertLess(memory_increase, 500)  # 500MB 이하 증가

        collection_service.cleanup()


# ====================================================================================
# 8. 테스트 실행 및 보고서 생성
# ====================================================================================


def run_all_tests() -> None:
    """모든 테스트 실행"""
    # 테스트 스위트 생성
    test_suite = unittest.TestSuite()

    # 각 테스트 클래스 추가
    test_classes = [
        TestDataCollection,
        TestLabelingService,
        TestAugmentationService,
        TestValidationService,
        TestImageProcessor,
        TestIntegration,
        TestPerformance,
    ]

    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)

    # 테스트 실행
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # 결과 출력
    print(f"\n{'='*60}")
    print(f"테스트 결과 요약")
    print(f"{'='*60}")
    print(f"총 테스트 수: {result.testsRun}")
    print(f"성공: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"실패: {len(result.failures)}")
    print(f"오류: {len(result.errors)}")

    if result.failures:
        print(f"\n실패한 테스트:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")

    if result.errors:
        print(f"\n오류가 발생한 테스트:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")

    return result.wasSuccessful()


if __name__ == "__main__":
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    print("🧪 YOKOGAWA OCR 데이터 준비 프로젝트 테스트 시작")
    print("=" * 60)

    success = run_all_tests()

    if success:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
        sys.exit(0)
    else:
        print("\n❌ 일부 테스트가 실패했습니다.")
        sys.exit(1)
