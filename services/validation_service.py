#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 검증 서비스 모듈

이 모듈은 데이터셋 검증, 품질 확인, 일관성 검사 등의 검증 기능을 제공합니다.
최종 데이터셋의 품질을 보장하고 OCR 모델 훈련에 적합한 데이터를 준비합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import json
import uuid
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import threading
import time

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report

from core.base_classes import BaseService, ValidationInterface
from core.exceptions import (
    ValidationError,
    DataIntegrityError,
    ProcessingError,
    ApplicationError,
    FileAccessError,
    ConfigurationError,
)
from config.settings import ApplicationConfig
from config.constants import (
    DATA_QUALITY_MIN_SCORE,
    DATA_COMPLETENESS_MIN_SCORE,
    DATA_CONSISTENCY_MIN_SCORE,
    VALIDATION_COMPLETENESS_WEIGHT,
    VALIDATION_ACCURACY_WEIGHT,
    VALIDATION_CONSISTENCY_WEIGHT,
    VALIDATION_MAX_RETRY_COUNT,
    VALIDATION_RETRY_DELAY_SECONDS,
    STATISTICS_CONFIDENCE_LEVEL,
    STATISTICS_SAMPLE_SIZE,
    STATISTICS_PRECISION_DIGITS,
    CROSS_VALIDATION_FOLDS,
    CROSS_VALIDATION_RANDOM_STATE,
    DATA_SPLIT_RATIOS,
    MIN_TRAIN_DATASET_SIZE,
    MIN_VALIDATION_DATASET_SIZE,
    MIN_TEST_DATASET_SIZE,
    DEFAULT_BATCH_SIZE,
    MAX_WORKER_THREADS,
)
from models.document_model import DocumentModel, DocumentStatus
from models.annotation_model import (
    AnnotationModel,
    DocumentAnnotation,
    AnnotationStatus,
)
from utils.logger_util import get_application_logger
from utils.file_handler import FileHandler


class DataQualityValidator:
    """
    데이터 품질 검증 클래스

    데이터의 완성도, 정확성, 일관성 등을 검증합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        DataQualityValidator 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("data_quality_validator")

        # 품질 임계값 설정
        self.quality_thresholds = {
            "completeness": DATA_COMPLETENESS_MIN_SCORE,
            "accuracy": DATA_QUALITY_MIN_SCORE,
            "consistency": DATA_CONSISTENCY_MIN_SCORE,
        }

        # 검증 통계
        self.validation_statistics = {
            "total_validated": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "error_categories": defaultdict(int),
        }

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

        self.logger.info("DataQualityValidator initialized")

    def validate_dataset_quality(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        데이터셋 품질 검증

        Args:
            dataset: 검증할 데이터셋

        Returns:
            Dict[str, Any]: 품질 검증 결과
        """
        try:
            self.logger.info(
                f"Starting dataset quality validation for {len(dataset)} items"
            )

            # 완성도 검증
            completeness_score = self._validate_completeness(dataset)

            # 정확성 검증
            accuracy_score = self._validate_accuracy(dataset)

            # 일관성 검증
            consistency_score = self._validate_consistency(dataset)

            # 종합 품질 점수 계산
            overall_score = self._calculate_overall_quality_score(
                completeness_score, accuracy_score, consistency_score
            )

            # 검증 결과 정리
            validation_result = {
                "overall_quality_score": overall_score,
                "completeness_score": completeness_score,
                "accuracy_score": accuracy_score,
                "consistency_score": consistency_score,
                "quality_passed": overall_score >= self.quality_thresholds["accuracy"],
                "detailed_results": {
                    "completeness_details": self._get_completeness_details(dataset),
                    "accuracy_details": self._get_accuracy_details(dataset),
                    "consistency_details": self._get_consistency_details(dataset),
                },
                "validation_timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
            }

            # 통계 업데이트
            self._update_validation_statistics(validation_result)

            self.logger.info(
                f"Dataset quality validation completed: {overall_score:.3f}"
            )
            return validation_result

        except Exception as e:
            self.logger.error(f"Dataset quality validation failed: {str(e)}")
            raise ValidationError(
                message=f"Dataset quality validation failed: {str(e)}",
                validation_type="dataset_quality",
                original_exception=e,
            )

    def _validate_completeness(self, dataset: List[Dict[str, Any]]) -> float:
        """완성도 검증"""
        try:
            if not dataset:
                return 0.0

            completed_items = 0
            total_items = len(dataset)

            for item in dataset:
                # 필수 필드 확인
                required_fields = ["document_id", "annotations", "status"]
                has_required_fields = all(field in item for field in required_fields)

                # 어노테이션 완성도 확인
                annotations_complete = self._check_annotations_completeness(
                    item.get("annotations", [])
                )

                if has_required_fields and annotations_complete:
                    completed_items += 1

            completeness_score = (
                completed_items / total_items if total_items > 0 else 0.0
            )
            return completeness_score

        except Exception as e:
            self.logger.error(f"Completeness validation failed: {str(e)}")
            return 0.0

    def _validate_accuracy(self, dataset: List[Dict[str, Any]]) -> float:
        """정확성 검증"""
        try:
            if not dataset:
                return 0.0

            accurate_items = 0
            total_items = len(dataset)

            for item in dataset:
                # 데이터 형식 검증
                format_valid = self._validate_data_format(item)

                # 어노테이션 정확성 검증
                annotations_accurate = self._check_annotations_accuracy(
                    item.get("annotations", [])
                )

                if format_valid and annotations_accurate:
                    accurate_items += 1

            accuracy_score = accurate_items / total_items if total_items > 0 else 0.0
            return accuracy_score

        except Exception as e:
            self.logger.error(f"Accuracy validation failed: {str(e)}")
            return 0.0

    def _validate_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """일관성 검증"""
        try:
            if not dataset:
                return 0.0

            # 스키마 일관성 검증
            schema_consistency = self._check_schema_consistency(dataset)

            # 어노테이션 일관성 검증
            annotation_consistency = self._check_annotation_consistency(dataset)

            # 명명 규칙 일관성 검증
            naming_consistency = self._check_naming_consistency(dataset)

            # 전체 일관성 점수 계산
            consistency_score = (
                schema_consistency + annotation_consistency + naming_consistency
            ) / 3
            return consistency_score

        except Exception as e:
            self.logger.error(f"Consistency validation failed: {str(e)}")
            return 0.0

    def _check_annotations_completeness(
        self, annotations: List[Dict[str, Any]]
    ) -> bool:
        """어노테이션 완성도 확인"""
        if not annotations:
            return False

        for annotation in annotations:
            # 필수 필드 확인
            required_fields = ["field_name", "bounding_box", "text_value"]
            if not all(field in annotation for field in required_fields):
                return False

            # 텍스트 값 확인
            if not annotation.get("text_value", "").strip():
                return False

            # 바운딩 박스 확인
            bbox = annotation.get("bounding_box", {})
            if not all(key in bbox for key in ["x", "y", "width", "height"]):
                return False

        return True

    def _check_annotations_accuracy(self, annotations: List[Dict[str, Any]]) -> bool:
        """어노테이션 정확성 확인"""
        if not annotations:
            return False

        for annotation in annotations:
            # 신뢰도 점수 확인
            confidence = annotation.get("confidence_score", 0.0)
            if confidence < 0.7:  # 70% 미만은 부정확으로 간주
                return False

            # 바운딩 박스 유효성 확인
            bbox = annotation.get("bounding_box", {})
            if any(bbox.get(key, 0) < 0 for key in ["x", "y", "width", "height"]):
                return False

        return True

    def _validate_data_format(self, item: Dict[str, Any]) -> bool:
        """데이터 형식 검증"""
        try:
            # 기본 필드 타입 검증
            if not isinstance(item.get("document_id"), str):
                return False

            if not isinstance(item.get("annotations"), list):
                return False

            # 상태 값 검증
            valid_statuses = ["pending", "in_progress", "completed", "validated"]
            if item.get("status") not in valid_statuses:
                return False

            return True

        except Exception:
            return False

    def _check_schema_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """스키마 일관성 검증"""
        if not dataset:
            return 0.0

        # 첫 번째 아이템을 기준 스키마로 사용
        base_schema = set(dataset[0].keys())

        consistent_items = 0
        for item in dataset:
            item_schema = set(item.keys())
            if item_schema == base_schema:
                consistent_items += 1

        return consistent_items / len(dataset)

    def _check_annotation_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """어노테이션 일관성 검증"""
        if not dataset:
            return 0.0

        # 어노테이션 필드 이름 일관성 확인
        field_names = set()
        for item in dataset:
            for annotation in item.get("annotations", []):
                field_names.add(annotation.get("field_name"))

        # 모든 문서에서 동일한 필드 이름 사용하는지 확인
        consistent_items = 0
        for item in dataset:
            item_field_names = set(
                annotation.get("field_name")
                for annotation in item.get("annotations", [])
            )
            if item_field_names.issubset(field_names):
                consistent_items += 1

        return consistent_items / len(dataset)

    def _check_naming_consistency(self, dataset: List[Dict[str, Any]]) -> float:
        """명명 규칙 일관성 검증"""
        if not dataset:
            return 0.0

        consistent_items = 0
        for item in dataset:
            # document_id 형식 확인 (UUID 형식)
            doc_id = item.get("document_id", "")
            if len(doc_id) == 36 and doc_id.count("-") == 4:
                consistent_items += 1

        return consistent_items / len(dataset)

    def _calculate_overall_quality_score(
        self, completeness: float, accuracy: float, consistency: float
    ) -> float:
        """종합 품질 점수 계산"""
        weights = {
            "completeness": VALIDATION_COMPLETENESS_WEIGHT,
            "accuracy": VALIDATION_ACCURACY_WEIGHT,
            "consistency": VALIDATION_CONSISTENCY_WEIGHT,
        }

        overall_score = (
            completeness * weights["completeness"]
            + accuracy * weights["accuracy"]
            + consistency * weights["consistency"]
        )

        return round(overall_score, STATISTICS_PRECISION_DIGITS)

    def _get_completeness_details(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """완성도 세부 정보"""
        total_items = len(dataset)
        completed_items = sum(
            1
            for item in dataset
            if self._check_annotations_completeness(item.get("annotations", []))
        )

        return {
            "total_items": total_items,
            "completed_items": completed_items,
            "completion_rate": (
                completed_items / total_items if total_items > 0 else 0.0
            ),
        }

    def _get_accuracy_details(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """정확성 세부 정보"""
        total_items = len(dataset)
        accurate_items = sum(
            1
            for item in dataset
            if self._check_annotations_accuracy(item.get("annotations", []))
        )

        return {
            "total_items": total_items,
            "accurate_items": accurate_items,
            "accuracy_rate": accurate_items / total_items if total_items > 0 else 0.0,
        }

    def _get_consistency_details(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """일관성 세부 정보"""
        schema_consistency = self._check_schema_consistency(dataset)
        annotation_consistency = self._check_annotation_consistency(dataset)
        naming_consistency = self._check_naming_consistency(dataset)

        return {
            "schema_consistency": schema_consistency,
            "annotation_consistency": annotation_consistency,
            "naming_consistency": naming_consistency,
            "overall_consistency": (
                schema_consistency + annotation_consistency + naming_consistency
            )
            / 3,
        }

    def _update_validation_statistics(self, validation_result: Dict[str, Any]) -> None:
        """검증 통계 업데이트"""
        with self._lock:
            self.validation_statistics["total_validated"] += 1

            if validation_result["quality_passed"]:
                self.validation_statistics["passed_validations"] += 1
            else:
                self.validation_statistics["failed_validations"] += 1

    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        with self._lock:
            return self.validation_statistics.copy()


class ConsistencyChecker:
    """
    일관성 검사 클래스

    데이터간 일관성, 어노테이션 일관성 등을 검사합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        ConsistencyChecker 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("consistency_checker")

        # 일관성 체크 규칙
        self.consistency_rules = {
            "annotation_overlap": self._check_annotation_overlap,
            "field_name_consistency": self._check_field_name_consistency,
            "data_type_consistency": self._check_data_type_consistency,
            "coordinate_consistency": self._check_coordinate_consistency,
        }

        self.logger.info("ConsistencyChecker initialized")

    def check_annotation_consistency(
        self, annotations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        어노테이션 일관성 검사

        Args:
            annotations: 검사할 어노테이션 목록

        Returns:
            List[str]: 일관성 오류 목록
        """
        try:
            errors = []

            for rule_name, rule_func in self.consistency_rules.items():
                try:
                    rule_errors = rule_func(annotations)
                    if rule_errors:
                        errors.extend(rule_errors)
                except Exception as e:
                    self.logger.error(
                        f"Consistency rule '{rule_name}' failed: {str(e)}"
                    )
                    errors.append(f"Consistency rule '{rule_name}' execution failed")

            return errors

        except Exception as e:
            self.logger.error(f"Annotation consistency check failed: {str(e)}")
            return [f"Consistency check failed: {str(e)}"]

    def _check_annotation_overlap(self, annotations: List[Dict[str, Any]]) -> List[str]:
        """어노테이션 겹침 검사"""
        errors = []

        for i, ann1 in enumerate(annotations):
            for j, ann2 in enumerate(annotations[i + 1 :], i + 1):
                bbox1 = ann1.get("bounding_box", {})
                bbox2 = ann2.get("bounding_box", {})

                if self._bounding_boxes_overlap(bbox1, bbox2):
                    errors.append(
                        f"Annotation overlap detected between annotation {i} and {j}"
                    )

        return errors

    def _check_field_name_consistency(
        self, annotations: List[Dict[str, Any]]
    ) -> List[str]:
        """필드 이름 일관성 검사"""
        errors = []

        # 동일한 필드 이름의 어노테이션들 그룹화
        field_groups = defaultdict(list)
        for i, annotation in enumerate(annotations):
            field_name = annotation.get("field_name", "")
            field_groups[field_name].append((i, annotation))

        # 각 그룹 내에서 일관성 검사
        for field_name, group in field_groups.items():
            if len(group) > 1:
                # 같은 필드 이름이 여러 개 있는 경우 경고
                indices = [str(i) for i, _ in group]
                errors.append(
                    f"Duplicate field name '{field_name}' found in annotations: {', '.join(indices)}"
                )

        return errors

    def _check_data_type_consistency(
        self, annotations: List[Dict[str, Any]]
    ) -> List[str]:
        """데이터 타입 일관성 검사"""
        errors = []

        for i, annotation in enumerate(annotations):
            # 필수 필드 타입 검사
            if not isinstance(annotation.get("field_name"), str):
                errors.append(f"Annotation {i}: field_name must be string")

            if not isinstance(annotation.get("text_value"), str):
                errors.append(f"Annotation {i}: text_value must be string")

            # 바운딩 박스 타입 검사
            bbox = annotation.get("bounding_box", {})
            for coord in ["x", "y", "width", "height"]:
                if coord not in bbox or not isinstance(bbox[coord], (int, float)):
                    errors.append(
                        f"Annotation {i}: bounding_box.{coord} must be number"
                    )

        return errors

    def _check_coordinate_consistency(
        self, annotations: List[Dict[str, Any]]
    ) -> List[str]:
        """좌표 일관성 검사"""
        errors = []

        for i, annotation in enumerate(annotations):
            bbox = annotation.get("bounding_box", {})

            # 좌표 값 유효성 검사
            x = bbox.get("x", 0)
            y = bbox.get("y", 0)
            width = bbox.get("width", 0)
            height = bbox.get("height", 0)

            if x < 0 or y < 0:
                errors.append(f"Annotation {i}: negative coordinates not allowed")

            if width <= 0 or height <= 0:
                errors.append(f"Annotation {i}: width and height must be positive")

            # 종횡비 검사
            if width > 0 and height > 0:
                aspect_ratio = width / height
                if aspect_ratio > 20 or aspect_ratio < 0.05:
                    errors.append(
                        f"Annotation {i}: unusual aspect ratio {aspect_ratio:.2f}"
                    )

        return errors

    def _bounding_boxes_overlap(
        self, bbox1: Dict[str, Any], bbox2: Dict[str, Any]
    ) -> bool:
        """바운딩 박스 겹침 확인"""
        try:
            x1, y1, w1, h1 = (
                bbox1.get("x", 0),
                bbox1.get("y", 0),
                bbox1.get("width", 0),
                bbox1.get("height", 0),
            )
            x2, y2, w2, h2 = (
                bbox2.get("x", 0),
                bbox2.get("y", 0),
                bbox2.get("width", 0),
                bbox2.get("height", 0),
            )

            # 겹침 검사
            if x1 < x2 + w2 and x1 + w1 > x2 and y1 < y2 + h2 and y1 + h1 > y2:
                return True

            return False

        except Exception:
            return False


class StatisticsGenerator:
    """
    통계 생성 클래스

    데이터셋의 다양한 통계 정보를 생성합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        StatisticsGenerator 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("statistics_generator")

        # 통계 설정
        self.confidence_level = STATISTICS_CONFIDENCE_LEVEL
        self.sample_size = STATISTICS_SAMPLE_SIZE
        self.precision_digits = STATISTICS_PRECISION_DIGITS

        self.logger.info("StatisticsGenerator initialized")

    def generate_dataset_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        데이터셋 통계 생성

        Args:
            dataset: 통계를 생성할 데이터셋

        Returns:
            Dict[str, Any]: 생성된 통계 정보
        """
        try:
            self.logger.info(
                f"Generating statistics for dataset with {len(dataset)} items"
            )

            # 기본 통계
            basic_stats = self._generate_basic_statistics(dataset)

            # 어노테이션 통계
            annotation_stats = self._generate_annotation_statistics(dataset)

            # 품질 통계
            quality_stats = self._generate_quality_statistics(dataset)

            # 분포 통계
            distribution_stats = self._generate_distribution_statistics(dataset)

            # 종합 통계
            overall_stats = {
                "basic_statistics": basic_stats,
                "annotation_statistics": annotation_stats,
                "quality_statistics": quality_stats,
                "distribution_statistics": distribution_stats,
                "generation_timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
            }

            self.logger.info("Dataset statistics generation completed")
            return overall_stats

        except Exception as e:
            self.logger.error(f"Statistics generation failed: {str(e)}")
            raise ProcessingError(
                message=f"Statistics generation failed: {str(e)}",
                processor_id="statistics_generator",
                processing_stage="statistics_generation",
                original_exception=e,
            )

    def _generate_basic_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """기본 통계 생성"""
        try:
            total_items = len(dataset)

            # 상태별 통계
            status_counts = Counter(item.get("status", "unknown") for item in dataset)

            # 문서 타입별 통계
            doc_type_counts = Counter(
                item.get("document_type", "unknown") for item in dataset
            )

            return {
                "total_items": total_items,
                "status_distribution": dict(status_counts),
                "document_type_distribution": dict(doc_type_counts),
                "unique_documents": len(
                    set(item.get("document_id", "") for item in dataset)
                ),
            }

        except Exception as e:
            self.logger.error(f"Basic statistics generation failed: {str(e)}")
            return {}

    def _generate_annotation_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """어노테이션 통계 생성"""
        try:
            all_annotations = []
            for item in dataset:
                all_annotations.extend(item.get("annotations", []))

            if not all_annotations:
                return {"total_annotations": 0}

            # 필드 이름 분포
            field_name_counts = Counter(
                ann.get("field_name", "") for ann in all_annotations
            )

            # 신뢰도 점수 통계
            confidence_scores = [
                ann.get("confidence_score", 0.0) for ann in all_annotations
            ]

            # 바운딩 박스 크기 통계
            bbox_areas = []
            for ann in all_annotations:
                bbox = ann.get("bounding_box", {})
                width = bbox.get("width", 0)
                height = bbox.get("height", 0)
                bbox_areas.append(width * height)

            return {
                "total_annotations": len(all_annotations),
                "field_name_distribution": dict(field_name_counts),
                "confidence_statistics": {
                    "mean": round(
                        statistics.mean(confidence_scores), self.precision_digits
                    ),
                    "median": round(
                        statistics.median(confidence_scores), self.precision_digits
                    ),
                    "std": (
                        round(
                            statistics.stdev(confidence_scores), self.precision_digits
                        )
                        if len(confidence_scores) > 1
                        else 0
                    ),
                    "min": round(min(confidence_scores), self.precision_digits),
                    "max": round(max(confidence_scores), self.precision_digits),
                },
                "bounding_box_area_statistics": {
                    "mean": round(statistics.mean(bbox_areas), self.precision_digits),
                    "median": round(
                        statistics.median(bbox_areas), self.precision_digits
                    ),
                    "std": (
                        round(statistics.stdev(bbox_areas), self.precision_digits)
                        if len(bbox_areas) > 1
                        else 0
                    ),
                    "min": round(min(bbox_areas), self.precision_digits),
                    "max": round(max(bbox_areas), self.precision_digits),
                },
            }

        except Exception as e:
            self.logger.error(f"Annotation statistics generation failed: {str(e)}")
            return {"total_annotations": 0}

    def _generate_quality_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """품질 통계 생성"""
        try:
            quality_scores = []
            for item in dataset:
                # 품질 점수 계산 (간단한 휴리스틱)
                annotations = item.get("annotations", [])
                if annotations:
                    avg_confidence = statistics.mean(
                        ann.get("confidence_score", 0.0) for ann in annotations
                    )
                    completeness = (
                        1.0
                        if all(ann.get("text_value", "").strip() for ann in annotations)
                        else 0.5
                    )
                    quality_score = (avg_confidence + completeness) / 2
                    quality_scores.append(quality_score)

            if not quality_scores:
                return {"average_quality": 0.0}

            return {
                "average_quality": round(
                    statistics.mean(quality_scores), self.precision_digits
                ),
                "quality_std": (
                    round(statistics.stdev(quality_scores), self.precision_digits)
                    if len(quality_scores) > 1
                    else 0
                ),
                "min_quality": round(min(quality_scores), self.precision_digits),
                "max_quality": round(max(quality_scores), self.precision_digits),
                "quality_distribution": {
                    "excellent": sum(1 for score in quality_scores if score >= 0.9),
                    "good": sum(1 for score in quality_scores if 0.7 <= score < 0.9),
                    "fair": sum(1 for score in quality_scores if 0.5 <= score < 0.7),
                    "poor": sum(1 for score in quality_scores if score < 0.5),
                },
            }

        except Exception as e:
            self.logger.error(f"Quality statistics generation failed: {str(e)}")
            return {"average_quality": 0.0}

    def _generate_distribution_statistics(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """분포 통계 생성"""
        try:
            # 어노테이션 수 분포
            annotation_counts = [len(item.get("annotations", [])) for item in dataset]

            # 텍스트 길이 분포
            text_lengths = []
            for item in dataset:
                for ann in item.get("annotations", []):
                    text_value = ann.get("text_value", "")
                    text_lengths.append(len(text_value))

            return {
                "annotations_per_document": {
                    "mean": round(
                        statistics.mean(annotation_counts), self.precision_digits
                    ),
                    "median": round(
                        statistics.median(annotation_counts), self.precision_digits
                    ),
                    "std": (
                        round(
                            statistics.stdev(annotation_counts), self.precision_digits
                        )
                        if len(annotation_counts) > 1
                        else 0
                    ),
                    "min": min(annotation_counts),
                    "max": max(annotation_counts),
                },
                "text_length_distribution": {
                    "mean": (
                        round(statistics.mean(text_lengths), self.precision_digits)
                        if text_lengths
                        else 0
                    ),
                    "median": (
                        round(statistics.median(text_lengths), self.precision_digits)
                        if text_lengths
                        else 0
                    ),
                    "std": (
                        round(statistics.stdev(text_lengths), self.precision_digits)
                        if len(text_lengths) > 1
                        else 0
                    ),
                    "min": min(text_lengths) if text_lengths else 0,
                    "max": max(text_lengths) if text_lengths else 0,
                },
            }

        except Exception as e:
            self.logger.error(f"Distribution statistics generation failed: {str(e)}")
            return {}


class ValidationService(BaseService, ValidationInterface):
    """
    검증 서비스 클래스

    데이터셋 검증, 품질 확인, 일관성 검사 등의 검증 기능을 제공합니다.
    BaseService와 ValidationInterface를 구현합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        ValidationService 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)

        # 컴포넌트 초기화
        self.file_handler = FileHandler(config)
        self.data_quality_validator = DataQualityValidator(config)
        self.consistency_checker = ConsistencyChecker(config)
        self.statistics_generator = StatisticsGenerator(config)

        # 검증 기준 설정
        self.validation_criteria = {
            "min_quality_score": DATA_QUALITY_MIN_SCORE,
            "min_completeness_score": DATA_COMPLETENESS_MIN_SCORE,
            "min_consistency_score": DATA_CONSISTENCY_MIN_SCORE,
            "min_dataset_size": MIN_TRAIN_DATASET_SIZE,
            "required_fields": ["document_id", "annotations", "status"],
        }

        # 상태 관리
        self.validated_datasets: List[Dict[str, Any]] = []
        self.validation_history: List[Dict[str, Any]] = []
        self.validation_errors: List[str] = []

        # 검증 통계
        self.validation_statistics = {
            "total_validations": 0,
            "passed_validations": 0,
            "failed_validations": 0,
            "average_quality_score": 0.0,
            "validation_duration": 0.0,
        }

        # 진행 상태
        self.validation_progress = 0.0
        self.current_operation: Optional[str] = None
        self.processing_errors: List[str] = []

        # 콜백 관리
        self.validation_callbacks: List[Callable] = []

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """
        서비스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("Initializing ValidationService")

            # 상태 초기화
            with self._lock:
                self.validated_datasets.clear()
                self.validation_history.clear()
                self.validation_errors.clear()
                self.processing_errors.clear()
                self.validation_progress = 0.0
                self.current_operation = None

            # 검증 기준 검증
            self._validate_criteria()

            self.logger.info("ValidationService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize ValidationService: {str(e)}")
            self._is_initialized = False
            return False

    def cleanup(self) -> None:
        """
        서비스 정리
        """
        try:
            self.logger.info("Cleaning up ValidationService")

            with self._lock:
                self.validated_datasets.clear()
                self.validation_history.clear()
                self.validation_errors.clear()
                self.processing_errors.clear()

            # 파일 핸들러 정리
            if hasattr(self.file_handler, "cleanup_temp_files"):
                self.file_handler.cleanup_temp_files()

            self.logger.info("ValidationService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during ValidationService cleanup: {str(e)}")

    def health_check(self) -> bool:
        """서비스 상태 확인"""
        try:
            # 초기화 상태 확인
            if not self.is_initialized():
                self.logger.warning("Service not initialized")
                return False
                
            # 기본 컴포넌트 확인
            if not hasattr(self, 'config') or self.config is None:
                self.logger.warning("Config is None")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False


    def validate_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        데이터셋 검증 (ValidationInterface 구현)

        Args:
            dataset: 검증할 데이터셋

        Returns:
            Dict[str, Any]: 검증 결과
        """
        try:
            self.logger.info(f"Starting dataset validation for {len(dataset)} items")

            start_time = time.time()

            with self._lock:
                self.current_operation = "dataset_validation"
                self.validation_progress = 0.0

            # 1. 기본 검증
            self._update_progress(0.1, "Performing basic validation")
            basic_validation = self._validate_basic_requirements(dataset)

            # 2. 데이터 품질 검증
            self._update_progress(0.3, "Validating data quality")
            quality_validation = self.data_quality_validator.validate_dataset_quality(
                dataset
            )

            # 3. 일관성 검증
            self._update_progress(0.5, "Checking consistency")
            consistency_validation = self._validate_consistency(dataset)

            # 4. 데이터 분할 검증
            self._update_progress(0.7, "Validating data split")
            split_validation = self._validate_data_split(dataset)

            # 5. 통계 생성
            self._update_progress(0.9, "Generating statistics")
            dataset_statistics = self.statistics_generator.generate_dataset_statistics(
                dataset
            )

            # 검증 결과 종합
            validation_result = {
                "validation_id": str(uuid.uuid4()),
                "validation_timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
                "validation_passed": (
                    basic_validation["passed"]
                    and quality_validation["quality_passed"]
                    and consistency_validation["consistency_passed"]
                    and split_validation["split_valid"]
                ),
                "basic_validation": basic_validation,
                "quality_validation": quality_validation,
                "consistency_validation": consistency_validation,
                "split_validation": split_validation,
                "dataset_statistics": dataset_statistics,
                "validation_duration": time.time() - start_time,
                "validation_criteria": self.validation_criteria,
            }

            # 검증 결과 저장
            self._save_validation_result(validation_result)

            # 통계 업데이트
            self._update_validation_statistics(validation_result)

            # 콜백 실행
            self._execute_validation_callbacks(validation_result)

            with self._lock:
                self.current_operation = None
                self.validation_progress = 1.0

            self.logger.info(
                f"Dataset validation completed: {'PASSED' if validation_result['validation_passed'] else 'FAILED'}"
            )
            return validation_result

        except Exception as e:
            self.logger.error(f"Dataset validation failed: {str(e)}")
            with self._lock:
                self.processing_errors.append(str(e))
                self.current_operation = None
            raise ValidationError(
                message=f"Dataset validation failed: {str(e)}",
                validation_type="dataset_validation",
                original_exception=e,
            )

    def get_validation_report(self) -> Dict[str, Any]:
        """
        검증 보고서 생성 (ValidationInterface 구현)

        Returns:
            Dict[str, Any]: 검증 보고서
        """
        try:
            self.logger.info("Generating validation report")

            with self._lock:
                # 최근 검증 결과
                recent_validations = (
                    self.validation_history[-10:] if self.validation_history else []
                )

                # 검증 통계
                validation_stats = self.validation_statistics.copy()

                # 오류 분석
                error_analysis = self._analyze_validation_errors()

                # 품질 트렌드
                quality_trends = self._analyze_quality_trends()

                report = {
                    "report_id": str(uuid.uuid4()),
                    "report_timestamp": datetime.now().isoformat(),
                    "validation_statistics": validation_stats,
                    "recent_validations": recent_validations,
                    "error_analysis": error_analysis,
                    "quality_trends": quality_trends,
                    "recommendations": self._generate_recommendations(),
                }

            self.logger.info("Validation report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"Validation report generation failed: {str(e)}")
            raise ProcessingError(
                message=f"Validation report generation failed: {str(e)}",
                processor_id=self.service_id,
                processing_stage="report_generation",
                original_exception=e,
            )

    def set_validation_criteria(self, criteria: Dict[str, Any]) -> None:
        """
        검증 기준 설정 (ValidationInterface 구현)

        Args:
            criteria: 검증 기준
        """
        try:
            self.logger.info(f"Setting validation criteria: {criteria}")

            # 기준 검증
            self._validate_criteria_format(criteria)

            # 기준 업데이트
            self.validation_criteria.update(criteria)

            self.logger.info("Validation criteria updated successfully")

        except Exception as e:
            self.logger.error(f"Failed to set validation criteria: {str(e)}")
            raise ValidationError(
                message=f"Failed to set validation criteria: {str(e)}",
                validation_type="criteria_setting",
                original_exception=e,
            )

    def validate_dataset_completeness(self, dataset: List[Dict[str, Any]]) -> bool:
        """
        데이터셋 완성도 검증

        Args:
            dataset: 검증할 데이터셋

        Returns:
            bool: 완성도 검증 결과
        """
        try:
            self.logger.info(
                f"Validating dataset completeness for {len(dataset)} items"
            )

            # 최소 데이터 크기 확인
            if len(dataset) < self.validation_criteria["min_dataset_size"]:
                self.logger.warning(
                    f"Dataset size {len(dataset)} is below minimum {self.validation_criteria['min_dataset_size']}"
                )
                return False

            # 필수 필드 확인
            required_fields = self.validation_criteria["required_fields"]
            incomplete_items = 0

            for item in dataset:
                if not all(field in item for field in required_fields):
                    incomplete_items += 1

            completeness_rate = (len(dataset) - incomplete_items) / len(dataset)

            self.logger.info(f"Dataset completeness rate: {completeness_rate:.3f}")
            return (
                completeness_rate >= self.validation_criteria["min_completeness_score"]
            )

        except Exception as e:
            self.logger.error(f"Dataset completeness validation failed: {str(e)}")
            return False

    def check_annotation_consistency(
        self, annotations: List[Dict[str, Any]]
    ) -> List[str]:
        """
        어노테이션 일관성 확인

        Args:
            annotations: 확인할 어노테이션 목록

        Returns:
            List[str]: 일관성 오류 목록
        """
        try:
            return self.consistency_checker.check_annotation_consistency(annotations)

        except Exception as e:
            self.logger.error(f"Annotation consistency check failed: {str(e)}")
            return [f"Consistency check failed: {str(e)}"]

    def generate_quality_report(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        품질 보고서 생성

        Args:
            dataset: 보고서를 생성할 데이터셋

        Returns:
            Dict[str, Any]: 품질 보고서
        """
        try:
            self.logger.info("Generating quality report")

            # 데이터 품질 검증
            quality_validation = self.data_quality_validator.validate_dataset_quality(
                dataset
            )

            # 통계 생성
            dataset_statistics = self.statistics_generator.generate_dataset_statistics(
                dataset
            )

            # 일관성 검사
            consistency_errors = []
            for item in dataset:
                annotations = item.get("annotations", [])
                item_errors = self.consistency_checker.check_annotation_consistency(
                    annotations
                )
                consistency_errors.extend(item_errors)

            quality_report = {
                "report_id": str(uuid.uuid4()),
                "report_timestamp": datetime.now().isoformat(),
                "dataset_size": len(dataset),
                "quality_validation": quality_validation,
                "dataset_statistics": dataset_statistics,
                "consistency_errors": consistency_errors,
                "quality_summary": {
                    "overall_quality": quality_validation["overall_quality_score"],
                    "completeness": quality_validation["completeness_score"],
                    "accuracy": quality_validation["accuracy_score"],
                    "consistency": quality_validation["consistency_score"],
                    "total_errors": len(consistency_errors),
                },
            }

            self.logger.info("Quality report generated successfully")
            return quality_report

        except Exception as e:
            self.logger.error(f"Quality report generation failed: {str(e)}")
            raise ProcessingError(
                message=f"Quality report generation failed: {str(e)}",
                processor_id=self.service_id,
                processing_stage="quality_report_generation",
                original_exception=e,
            )

    def validate_data_split_ratios(self, dataset: List[Dict[str, Any]]) -> bool:
        """
        데이터 분할 비율 검증

        Args:
            dataset: 검증할 데이터셋

        Returns:
            bool: 분할 비율 검증 결과
        """
        try:
            self.logger.info("Validating data split ratios")

            total_size = len(dataset)

            # 각 분할의 최소 크기 확인
            train_size = int(total_size * DATA_SPLIT_RATIOS["train"])
            validation_size = int(total_size * DATA_SPLIT_RATIOS["validation"])
            test_size = int(total_size * DATA_SPLIT_RATIOS["test"])

            # 최소 크기 요구사항 확인
            if train_size < MIN_TRAIN_DATASET_SIZE:
                self.logger.warning(
                    f"Training set size {train_size} is below minimum {MIN_TRAIN_DATASET_SIZE}"
                )
                return False

            if validation_size < MIN_VALIDATION_DATASET_SIZE:
                self.logger.warning(
                    f"Validation set size {validation_size} is below minimum {MIN_VALIDATION_DATASET_SIZE}"
                )
                return False

            if test_size < MIN_TEST_DATASET_SIZE:
                self.logger.warning(
                    f"Test set size {test_size} is below minimum {MIN_TEST_DATASET_SIZE}"
                )
                return False

            # 분할 비율 합계 확인
            total_ratio = sum(DATA_SPLIT_RATIOS.values())
            if abs(total_ratio - 1.0) > 0.001:
                self.logger.warning(f"Split ratios sum to {total_ratio}, expected 1.0")
                return False

            self.logger.info("Data split ratios validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Data split ratios validation failed: {str(e)}")
            return False

    def perform_cross_validation(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        교차 검증 수행

        Args:
            annotations: 교차 검증할 어노테이션 목록

        Returns:
            Dict[str, float]: 교차 검증 결과
        """
        try:
            self.logger.info(
                f"Performing cross-validation with {len(annotations)} annotations"
            )

            if len(annotations) < CROSS_VALIDATION_FOLDS:
                self.logger.warning("Not enough data for cross-validation")
                return {"cv_score": 0.0, "cv_std": 0.0}

            # 간단한 교차 검증 구현 (품질 점수 기반)
            quality_scores = []

            # 데이터를 폴드로 분할
            fold_size = len(annotations) // CROSS_VALIDATION_FOLDS

            for fold in range(CROSS_VALIDATION_FOLDS):
                start_idx = fold * fold_size
                end_idx = (
                    start_idx + fold_size
                    if fold < CROSS_VALIDATION_FOLDS - 1
                    else len(annotations)
                )

                fold_annotations = annotations[start_idx:end_idx]

                # 폴드별 품질 점수 계산
                fold_quality = self._calculate_fold_quality(fold_annotations)
                quality_scores.append(fold_quality)

            cv_score = statistics.mean(quality_scores)
            cv_std = (
                statistics.stdev(quality_scores) if len(quality_scores) > 1 else 0.0
            )

            cross_validation_result = {
                "cv_score": round(cv_score, self.statistics_generator.precision_digits),
                "cv_std": round(cv_std, self.statistics_generator.precision_digits),
                "fold_scores": [
                    round(score, self.statistics_generator.precision_digits)
                    for score in quality_scores
                ],
                "folds": CROSS_VALIDATION_FOLDS,
            }

            self.logger.info(
                f"Cross-validation completed: CV score = {cv_score:.3f} ± {cv_std:.3f}"
            )
            return cross_validation_result

        except Exception as e:
            self.logger.error(f"Cross-validation failed: {str(e)}")
            return {"cv_score": 0.0, "cv_std": 0.0, "error": str(e)}

    def _validate_basic_requirements(
        self, dataset: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """기본 요구사항 검증"""
        try:
            errors = []

            # 최소 데이터 크기 확인
            if len(dataset) < self.validation_criteria["min_dataset_size"]:
                errors.append(
                    f"Dataset size {len(dataset)} is below minimum {self.validation_criteria['min_dataset_size']}"
                )

            # 필수 필드 확인
            required_fields = self.validation_criteria["required_fields"]
            missing_fields_count = 0

            for i, item in enumerate(dataset):
                for field in required_fields:
                    if field not in item:
                        missing_fields_count += 1
                        if missing_fields_count <= 10:  # 최대 10개까지만 로깅
                            errors.append(f"Item {i}: Missing required field '{field}'")

            return {
                "passed": len(errors) == 0,
                "errors": errors,
                "total_items": len(dataset),
                "missing_fields_count": missing_fields_count,
            }

        except Exception as e:
            self.logger.error(f"Basic requirements validation failed: {str(e)}")
            return {"passed": False, "errors": [str(e)]}

    def _validate_consistency(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """일관성 검증"""
        try:
            all_errors = []

            for i, item in enumerate(dataset):
                annotations = item.get("annotations", [])
                item_errors = self.consistency_checker.check_annotation_consistency(
                    annotations
                )

                # 아이템 인덱스 추가
                prefixed_errors = [f"Item {i}: {error}" for error in item_errors]
                all_errors.extend(prefixed_errors)

            consistency_score = max(0.0, 1.0 - len(all_errors) / len(dataset))

            return {
                "consistency_passed": consistency_score
                >= self.validation_criteria["min_consistency_score"],
                "consistency_score": consistency_score,
                "total_errors": len(all_errors),
                "errors": all_errors[:50],  # 최대 50개까지만 반환
            }

        except Exception as e:
            self.logger.error(f"Consistency validation failed: {str(e)}")
            return {
                "consistency_passed": False,
                "consistency_score": 0.0,
                "errors": [str(e)],
            }

    def _validate_data_split(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """데이터 분할 검증"""
        try:
            split_valid = self.validate_data_split_ratios(dataset)

            total_size = len(dataset)
            train_size = int(total_size * DATA_SPLIT_RATIOS["train"])
            validation_size = int(total_size * DATA_SPLIT_RATIOS["validation"])
            test_size = int(total_size * DATA_SPLIT_RATIOS["test"])

            return {
                "split_valid": split_valid,
                "total_size": total_size,
                "train_size": train_size,
                "validation_size": validation_size,
                "test_size": test_size,
                "split_ratios": DATA_SPLIT_RATIOS,
            }

        except Exception as e:
            self.logger.error(f"Data split validation failed: {str(e)}")
            return {"split_valid": False, "error": str(e)}

    def _calculate_fold_quality(self, annotations: List[Dict[str, Any]]) -> float:
        """폴드 품질 계산"""
        try:
            if not annotations:
                return 0.0

            # 신뢰도 점수 평균
            confidence_scores = [
                ann.get("confidence_score", 0.0) for ann in annotations
            ]
            avg_confidence = statistics.mean(confidence_scores)

            # 완성도 점수
            complete_annotations = sum(
                1 for ann in annotations if ann.get("text_value", "").strip()
            )
            completeness = complete_annotations / len(annotations)

            # 종합 품질 점수
            quality_score = (avg_confidence + completeness) / 2
            return quality_score

        except Exception as e:
            self.logger.error(f"Fold quality calculation failed: {str(e)}")
            return 0.0

    def _validate_criteria(self) -> None:
        """검증 기준 검증"""
        required_criteria = [
            "min_quality_score",
            "min_completeness_score",
            "min_consistency_score",
        ]

        for criterion in required_criteria:
            if criterion not in self.validation_criteria:
                raise ConfigurationError(f"Missing validation criterion: {criterion}")

            value = self.validation_criteria[criterion]
            if not isinstance(value, (int, float)) or not (0.0 <= value <= 1.0):
                raise ConfigurationError(
                    f"Invalid validation criterion value: {criterion} = {value}"
                )

    def _validate_criteria_format(self, criteria: Dict[str, Any]) -> None:
        """검증 기준 형식 검증"""
        for key, value in criteria.items():
            if key.endswith("_score") and isinstance(value, (int, float)):
                if not (0.0 <= value <= 1.0):
                    raise ValidationError(
                        f"Score criteria must be between 0.0 and 1.0: {key} = {value}"
                    )
            elif key.endswith("_size") and isinstance(value, int):
                if value < 1:
                    raise ValidationError(
                        f"Size criteria must be positive: {key} = {value}"
                    )

    def _save_validation_result(self, validation_result: Dict[str, Any]) -> None:
        """검증 결과 저장"""
        try:
            with self._lock:
                self.validation_history.append(validation_result)

                # 최대 100개까지만 유지
                if len(self.validation_history) > 100:
                    self.validation_history = self.validation_history[-100:]

                if validation_result["validation_passed"]:
                    self.validated_datasets.append(validation_result)

        except Exception as e:
            self.logger.error(f"Failed to save validation result: {str(e)}")

    def _update_validation_statistics(self, validation_result: Dict[str, Any]) -> None:
        """검증 통계 업데이트"""
        try:
            with self._lock:
                self.validation_statistics["total_validations"] += 1

                if validation_result["validation_passed"]:
                    self.validation_statistics["passed_validations"] += 1
                else:
                    self.validation_statistics["failed_validations"] += 1

                # 평균 품질 점수 업데이트
                quality_score = validation_result["quality_validation"][
                    "overall_quality_score"
                ]
                current_avg = self.validation_statistics["average_quality_score"]
                total_validations = self.validation_statistics["total_validations"]

                new_avg = (
                    current_avg * (total_validations - 1) + quality_score
                ) / total_validations
                self.validation_statistics["average_quality_score"] = new_avg

                # 검증 시간 업데이트
                duration = validation_result["validation_duration"]
                self.validation_statistics["validation_duration"] = duration

        except Exception as e:
            self.logger.error(f"Failed to update validation statistics: {str(e)}")

    def _analyze_validation_errors(self) -> Dict[str, Any]:
        """검증 오류 분석"""
        try:
            error_categories = defaultdict(int)

            for validation in self.validation_history:
                if not validation["validation_passed"]:
                    # 기본 검증 오류
                    for error in validation["basic_validation"].get("errors", []):
                        error_categories["basic_validation"] += 1

                    # 일관성 검증 오류
                    for error in validation["consistency_validation"].get("errors", []):
                        error_categories["consistency_validation"] += 1

            return {
                "error_categories": dict(error_categories),
                "total_errors": sum(error_categories.values()),
                "most_common_error": (
                    max(error_categories.items(), key=lambda x: x[1])[0]
                    if error_categories
                    else None
                ),
            }

        except Exception as e:
            self.logger.error(f"Error analysis failed: {str(e)}")
            return {"error_categories": {}, "total_errors": 0}

    def _analyze_quality_trends(self) -> Dict[str, Any]:
        """품질 트렌드 분석"""
        try:
            if len(self.validation_history) < 2:
                return {"trend": "insufficient_data"}

            # 최근 품질 점수들
            recent_scores = [
                validation["quality_validation"]["overall_quality_score"]
                for validation in self.validation_history[-10:]
            ]

            # 트렌드 계산
            if len(recent_scores) >= 2:
                trend = (
                    "improving" if recent_scores[-1] > recent_scores[0] else "declining"
                )
                avg_score = statistics.mean(recent_scores)

                return {
                    "trend": trend,
                    "recent_average": round(
                        avg_score, self.statistics_generator.precision_digits
                    ),
                    "score_range": [min(recent_scores), max(recent_scores)],
                    "data_points": len(recent_scores),
                }

            return {"trend": "stable"}

        except Exception as e:
            self.logger.error(f"Quality trend analysis failed: {str(e)}")
            return {"trend": "analysis_failed"}

    def _generate_recommendations(self) -> List[str]:
        """권장사항 생성"""
        try:
            recommendations = []

            # 검증 통계 기반 권장사항
            stats = self.validation_statistics

            if stats["total_validations"] > 0:
                pass_rate = stats["passed_validations"] / stats["total_validations"]

                if pass_rate < 0.7:
                    recommendations.append(
                        "검증 통과율이 낮습니다. 데이터 품질 개선이 필요합니다."
                    )

                if stats["average_quality_score"] < 0.8:
                    recommendations.append(
                        "평균 품질 점수가 낮습니다. 어노테이션 정확성을 확인하세요."
                    )

            # 오류 분석 기반 권장사항
            error_analysis = self._analyze_validation_errors()
            if error_analysis["total_errors"] > 0:
                recommendations.append(
                    "검증 오류가 많이 발생했습니다. 데이터 전처리 과정을 검토하세요."
                )

            # 기본 권장사항
            if not recommendations:
                recommendations.append(
                    "데이터 품질이 양호합니다. 정기적인 검증을 계속 수행하세요."
                )

            return recommendations

        except Exception as e:
            self.logger.error(f"Recommendations generation failed: {str(e)}")
            return ["권장사항 생성 중 오류가 발생했습니다."]

    def _update_progress(self, progress: float, operation: str) -> None:
        """진행률 업데이트"""
        with self._lock:
            self.validation_progress = progress
            self.current_operation = operation

        self.logger.debug(f"Progress updated: {progress:.1%} - {operation}")

    def _execute_validation_callbacks(self, validation_result: Dict[str, Any]) -> None:
        """검증 콜백 실행"""
        with self._lock:
            callbacks = self.validation_callbacks.copy()

        for callback in callbacks:
            try:
                callback(validation_result)
            except Exception as e:
                self.logger.error(f"Validation callback execution failed: {str(e)}")

    def get_validation_statistics(self) -> Dict[str, Any]:
        """검증 통계 반환"""
        with self._lock:
            return {
                **self.validation_statistics,
                "validation_history_count": len(self.validation_history),
                "validated_datasets_count": len(self.validated_datasets),
                "processing_errors_count": len(self.processing_errors),
                "service_id": self.service_id,
            }

    def get_validation_progress(self) -> Dict[str, Any]:
        """검증 진행 상황 반환"""
        with self._lock:
            return {
                "progress": self.validation_progress,
                "current_operation": self.current_operation,
                "validation_history_count": len(self.validation_history),
                "validated_datasets_count": len(self.validated_datasets),
                "processing_errors_count": len(self.processing_errors),
            }

    def register_validation_callback(self, callback: Callable) -> None:
        """검증 콜백 등록"""
        with self._lock:
            self.validation_callbacks.append(callback)

        self.logger.debug(f"Validation callback registered: {callback.__name__}")

    @classmethod
    def create_with_dependencies(cls, container) -> "ValidationService":
        """
        의존성 컨테이너를 사용한 팩토리 메서드

        Args:
            container: 의존성 컨테이너

        Returns:
            ValidationService: 생성된 서비스 인스턴스
        """
        return cls(
            config=container.get_service("config"),
            logger=container.get_service("logger"),
        )


# 모듈 수준 유틸리티 함수들
def create_validation_service(config: ApplicationConfig) -> ValidationService:
    """
    검증 서비스 생성 함수

    Args:
        config: 애플리케이션 설정

    Returns:
        ValidationService: 생성된 서비스 인스턴스
    """
    logger = get_application_logger("validation_service")
    service = ValidationService(config, logger)

    if not service.initialize():
        raise ProcessingError("Failed to initialize ValidationService")

    return service


if __name__ == "__main__":
    # 검증 서비스 테스트
    print("YOKOGAWA OCR 검증 서비스 테스트")
    print("=" * 50)

    try:
        # 설정 로드
        from config.settings import load_configuration

        config = load_configuration()

        # 서비스 생성
        service = create_validation_service(config)

        # 상태 확인
        if service.health_check():
            print("✅ 검증 서비스 정상 동작")
        else:
            print("❌ 검증 서비스 상태 이상")

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
                        "text_value": "Purchase Order",
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
                        "text_value": "YOKOGAWA Corp",
                        "confidence_score": 0.92,
                    }
                ],
            },
        ]

        # 데이터셋 검증 테스트
        validation_result = service.validate_dataset(test_dataset)
        print(
            f"📊 검증 결과: {'통과' if validation_result['validation_passed'] else '실패'}"
        )

        # 품질 보고서 생성
        quality_report = service.generate_quality_report(test_dataset)
        print(
            f"📈 품질 점수: {quality_report['quality_summary']['overall_quality']:.3f}"
        )

        # 통계 정보 출력
        statistics = service.get_validation_statistics()
        print(f"📊 검증 통계: {statistics}")

        # 정리
        service.cleanup()

    except Exception as e:
        print(f"❌ 검증 서비스 테스트 실패: {e}")

    print("\n🎯 검증 서비스 구현이 완료되었습니다!")
