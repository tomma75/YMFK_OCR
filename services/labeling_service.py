#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 라벨링 서비스 모듈

이 모듈은 문서 어노테이션 및 라벨링 기능을 제공합니다.
어노테이션 세션 관리, 품질 제어, 템플릿 관리 등을 포함합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Set, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from core.base_classes import BaseService, LabelingInterface
from core.exceptions import (
    LabelingError,
    AnnotationValidationError,
    ProcessingError,
    ValidationError,
    FileAccessError,
    FileFormatError,
)
from config.settings import ApplicationConfig
from config.constants import (
    ANNOTATION_FIELD_TYPES,
    ANNOTATION_QUALITY_THRESHOLD,
    ANNOTATION_CONFIDENCE_THRESHOLD,
    ANNOTATION_COMPLETENESS_THRESHOLD,
    ANNOTATION_SESSION_TIMEOUT_MINUTES,
    ANNOTATION_AUTO_SAVE_INTERVAL_SECONDS,
    ANNOTATION_MAX_ACTIVE_SESSIONS,
    DEFAULT_BATCH_SIZE,
    MAX_WORKER_THREADS,
)
from models.document_model import DocumentModel, DocumentStatus
from models.annotation_model import (
    AnnotationModel,
    FieldAnnotation,
    BoundingBox,
    AnnotationType,
    AnnotationStatus,
    DocumentAnnotation,
)
from utils.logger_util import get_application_logger
from utils.file_handler import FileHandler


class AnnotationManager:
    """
    어노테이션 관리자 클래스

    개별 어노테이션의 생성, 수정, 삭제 및 검증을 관리합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        AnnotationManager 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("annotation_manager")
        self.annotation_cache: Dict[str, AnnotationModel] = {}
        self.field_templates: Dict[str, Dict[str, Any]] = {}
        self.validation_rules: Dict[str, Callable] = {}

        # 통계 정보
        self.annotations_created = 0
        self.annotations_validated = 0
        self.annotations_rejected = 0

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

        self.logger.info("AnnotationManager initialized")

    def create_annotation(
        self,
        document_id: str,
        page_number: int,
        annotation_type: AnnotationType = AnnotationType.TEXT,
    ) -> AnnotationModel:
        """
        새로운 어노테이션 생성

        Args:
            document_id: 문서 ID
            page_number: 페이지 번호
            annotation_type: 어노테이션 타입

        Returns:
            AnnotationModel: 생성된 어노테이션 모델
        """
        try:
            annotation = AnnotationModel(
                document_id=document_id,
                page_number=page_number,
                annotation_type=annotation_type,
            )

            with self._lock:
                self.annotation_cache[annotation.annotation_id] = annotation
                self.annotations_created += 1

            self.logger.info(f"Annotation created: {annotation.annotation_id}")
            return annotation

        except Exception as e:
            self.logger.error(f"Failed to create annotation: {str(e)}")
            raise LabelingError(
                message=f"Annotation creation failed: {str(e)}",
                document_id=document_id,
                original_exception=e,
            )

    def add_field_annotation(
        self,
        annotation_id: str,
        field_name: str,
        field_type: AnnotationType,
        bounding_box: BoundingBox,
        text_value: str,
    ) -> FieldAnnotation:
        """
        필드 어노테이션 추가

        Args:
            annotation_id: 어노테이션 ID
            field_name: 필드 이름
            field_type: 필드 타입
            bounding_box: 바운딩 박스
            text_value: 텍스트 값

        Returns:
            FieldAnnotation: 생성된 필드 어노테이션
        """
        try:
            annotation = self.get_annotation(annotation_id)
            if not annotation:
                raise LabelingError(
                    message=f"Annotation not found: {annotation_id}",
                    document_id=annotation_id,
                )

            field_annotation = FieldAnnotation(
                field_id=str(uuid.uuid4()),
                field_name=field_name,
                field_type=field_type,
                bounding_box=bounding_box,
                text_value=text_value,
            )

            # 필드 검증
            if not self._validate_field_annotation(field_annotation):
                raise AnnotationValidationError(
                    message=f"Field annotation validation failed: {field_name}",
                    annotation_id=annotation_id,
                    validation_failures=field_annotation.validation_errors,
                )

            annotation.add_field_annotation(field_annotation)

            self.logger.info(f"Field annotation added: {field_name} to {annotation_id}")
            return field_annotation

        except Exception as e:
            self.logger.error(f"Failed to add field annotation: {str(e)}")
            raise LabelingError(
                message=f"Field annotation addition failed: {str(e)}",
                document_id=annotation_id,
                original_exception=e,
            )

    def get_annotation(self, annotation_id: str) -> Optional[AnnotationModel]:
        """
        어노테이션 조회

        Args:
            annotation_id: 어노테이션 ID

        Returns:
            Optional[AnnotationModel]: 어노테이션 모델 (없으면 None)
        """
        with self._lock:
            return self.annotation_cache.get(annotation_id)

    def update_annotation(self, annotation_id: str, updates: Dict[str, Any]) -> bool:
        """
        어노테이션 업데이트

        Args:
            annotation_id: 어노테이션 ID
            updates: 업데이트할 정보

        Returns:
            bool: 업데이트 성공 여부
        """
        try:
            annotation = self.get_annotation(annotation_id)
            if not annotation:
                return False

            for key, value in updates.items():
                if hasattr(annotation, key):
                    setattr(annotation, key, value)

            annotation.set_modified_time()

            self.logger.info(f"Annotation updated: {annotation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to update annotation {annotation_id}: {str(e)}")
            return False

    def validate_annotation(self, annotation_id: str) -> bool:
        """
        어노테이션 검증

        Args:
            annotation_id: 어노테이션 ID

        Returns:
            bool: 검증 성공 여부
        """
        try:
            annotation = self.get_annotation(annotation_id)
            if not annotation:
                return False

            # 기본 검증
            if not annotation.validate():
                with self._lock:
                    self.annotations_rejected += 1
                return False

            # 추가 검증 규칙 적용
            for rule_name, rule_func in self.validation_rules.items():
                if not rule_func(annotation):
                    self.logger.warning(
                        f"Validation rule '{rule_name}' failed for {annotation_id}"
                    )
                    with self._lock:
                        self.annotations_rejected += 1
                    return False

            annotation.set_validation_status(True)
            with self._lock:
                self.annotations_validated += 1

            self.logger.info(f"Annotation validated: {annotation_id}")
            return True

        except Exception as e:
            self.logger.error(f"Annotation validation failed: {str(e)}")
            return False

    def _validate_field_annotation(self, field_annotation: FieldAnnotation) -> bool:
        """
        필드 어노테이션 검증

        Args:
            field_annotation: 필드 어노테이션

        Returns:
            bool: 검증 결과
        """
        return field_annotation.validate_field_value()

    def get_statistics(self) -> Dict[str, Any]:
        """
        어노테이션 관리 통계 반환

        Returns:
            Dict[str, Any]: 통계 정보
        """
        with self._lock:
            return {
                "annotations_created": self.annotations_created,
                "annotations_validated": self.annotations_validated,
                "annotations_rejected": self.annotations_rejected,
                "annotations_in_cache": len(self.annotation_cache),
                "validation_rules_count": len(self.validation_rules),
                "field_templates_count": len(self.field_templates),
            }


class QualityController:
    """
    품질 제어 클래스

    어노테이션 품질 검사 및 품질 점수 계산을 관리합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        QualityController 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("quality_controller")
        self.quality_thresholds = {
            "confidence": ANNOTATION_CONFIDENCE_THRESHOLD,
            "completeness": ANNOTATION_COMPLETENESS_THRESHOLD,
            "overall": ANNOTATION_QUALITY_THRESHOLD,
        }

        # 품질 메트릭
        self.quality_metrics: Dict[str, List[float]] = {
            "confidence_scores": [],
            "completeness_scores": [],
            "consistency_scores": [],
        }

        self.logger.info("QualityController initialized")

    def calculate_quality_score(self, annotation: AnnotationModel) -> float:
        """
        어노테이션 품질 점수 계산

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 품질 점수 (0.0 ~ 1.0)
        """
        try:
            # 신뢰도 점수 계산
            confidence_score = self._calculate_confidence_score(annotation)

            # 완성도 점수 계산
            completeness_score = self._calculate_completeness_score(annotation)

            # 일관성 점수 계산
            consistency_score = self._calculate_consistency_score(annotation)

            # 가중 평균 계산
            weights = {"confidence": 0.4, "completeness": 0.4, "consistency": 0.2}
            overall_score = (
                confidence_score * weights["confidence"]
                + completeness_score * weights["completeness"]
                + consistency_score * weights["consistency"]
            )

            # 메트릭 업데이트
            self.quality_metrics["confidence_scores"].append(confidence_score)
            self.quality_metrics["completeness_scores"].append(completeness_score)
            self.quality_metrics["consistency_scores"].append(consistency_score)

            annotation.set_quality_score(overall_score)

            self.logger.debug(
                f"Quality score calculated: {overall_score:.3f} for {annotation.annotation_id}"
            )
            return overall_score

        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {str(e)}")
            return 0.0

    def _calculate_confidence_score(self, annotation: AnnotationModel) -> float:
        """
        신뢰도 점수 계산

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 신뢰도 점수
        """
        if not annotation.field_annotations:
            return 0.0

        confidence_sum = sum(
            field.confidence_score for field in annotation.field_annotations.values()
        )
        return confidence_sum / len(annotation.field_annotations)

    def _calculate_completeness_score(self, annotation: AnnotationModel) -> float:
        """
        완성도 점수 계산

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 완성도 점수
        """
        if not annotation.field_annotations:
            return 0.0

        total_fields = len(annotation.field_annotations)
        completed_fields = sum(
            1
            for field in annotation.field_annotations.values()
            if field.is_validated and field.text_value.strip()
        )

        return completed_fields / total_fields if total_fields > 0 else 0.0

    def _calculate_consistency_score(self, annotation: AnnotationModel) -> float:
        """
        일관성 점수 계산

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 일관성 점수
        """
        try:
            # 바운딩 박스 겹침 검사
            overlap_penalty = self._check_bounding_box_overlaps(annotation)

            # 필드 타입 일관성 검사
            type_consistency = self._check_field_type_consistency(annotation)

            # 텍스트 형식 일관성 검사
            format_consistency = self._check_text_format_consistency(annotation)

            # 전체 일관성 점수
            consistency_score = (
                type_consistency + format_consistency
            ) / 2 - overlap_penalty

            return max(0.0, min(1.0, consistency_score))

        except Exception as e:
            self.logger.error(f"Consistency score calculation failed: {str(e)}")
            return 0.5  # 기본값

    def _check_bounding_box_overlaps(self, annotation: AnnotationModel) -> float:
        """
        바운딩 박스 겹침 검사

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 겹침 페널티 (0.0 ~ 1.0)
        """
        field_annotations = list(annotation.field_annotations.values())
        overlap_count = 0
        total_pairs = 0

        for i in range(len(field_annotations)):
            for j in range(i + 1, len(field_annotations)):
                bbox1 = field_annotations[i].bounding_box
                bbox2 = field_annotations[j].bounding_box

                if bbox1.intersects_with(bbox2):
                    overlap_count += 1
                total_pairs += 1

        return overlap_count / total_pairs if total_pairs > 0 else 0.0

    def _check_field_type_consistency(self, annotation: AnnotationModel) -> float:
        """
        필드 타입 일관성 검사

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 타입 일관성 점수
        """
        consistent_fields = 0
        total_fields = len(annotation.field_annotations)

        for field in annotation.field_annotations.values():
            if field.field_type.value in ANNOTATION_FIELD_TYPES:
                consistent_fields += 1

        return consistent_fields / total_fields if total_fields > 0 else 1.0

    def _check_text_format_consistency(self, annotation: AnnotationModel) -> float:
        """
        텍스트 형식 일관성 검사

        Args:
            annotation: 어노테이션 모델

        Returns:
            float: 형식 일관성 점수
        """
        consistent_fields = 0
        total_fields = len(annotation.field_annotations)

        for field in annotation.field_annotations.values():
            if field.normalized_value is not None:
                consistent_fields += 1

        return consistent_fields / total_fields if total_fields > 0 else 1.0

    def check_quality_threshold(self, annotation: AnnotationModel) -> bool:
        """
        품질 임계값 검사

        Args:
            annotation: 어노테이션 모델

        Returns:
            bool: 품질 기준 통과 여부
        """
        quality_score = self.calculate_quality_score(annotation)
        return quality_score >= self.quality_thresholds["overall"]

    def get_quality_report(self) -> Dict[str, Any]:
        """
        품질 리포트 생성

        Returns:
            Dict[str, Any]: 품질 리포트
        """
        try:
            report = {"thresholds": self.quality_thresholds, "metrics_summary": {}}

            for metric_name, scores in self.quality_metrics.items():
                if scores:
                    report["metrics_summary"][metric_name] = {
                        "count": len(scores),
                        "average": sum(scores) / len(scores),
                        "min": min(scores),
                        "max": max(scores),
                    }
                else:
                    report["metrics_summary"][metric_name] = {
                        "count": 0,
                        "average": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }

            return report

        except Exception as e:
            self.logger.error(f"Quality report generation failed: {str(e)}")
            return {"error": str(e)}


class LabelingSessionManager:
    """
    라벨링 세션 관리자 클래스

    사용자 라벨링 세션의 생성, 관리, 타임아웃 처리를 담당합니다.
    """

    def __init__(self, config: ApplicationConfig):
        """
        LabelingSessionManager 초기화

        Args:
            config: 애플리케이션 설정 객체
        """
        self.config = config
        self.logger = get_application_logger("labeling_session_manager")
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(minutes=ANNOTATION_SESSION_TIMEOUT_MINUTES)
        self.auto_save_interval = ANNOTATION_AUTO_SAVE_INTERVAL_SECONDS
       
        self._lock = threading.Lock()
        
        # 세션 정리 스레드
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_expired_sessions, daemon=True
        )
        self.cleanup_thread.start()

        self.logger.info("LabelingSessionManager initialized")

    def create_session(self, document_path: str, annotator_id: str = "default") -> str:
        """
        새로운 라벨링 세션 생성

        Args:
            document_path: 문서 파일 경로
            annotator_id: 어노테이터 ID

        Returns:
            str: 세션 ID
        """
        try:
            # 활성 세션 수 확인
            with self._lock:
                if len(self.active_sessions) >= ANNOTATION_MAX_ACTIVE_SESSIONS:
                    raise LabelingError(
                        message=f"Maximum active sessions limit reached: {ANNOTATION_MAX_ACTIVE_SESSIONS}",
                        session_id=None,
                    )

            session_id = str(uuid.uuid4())
            session_data = {
                "session_id": session_id,
                "document_path": document_path,
                "annotator_id": annotator_id,
                "created_at": datetime.now(),
                "last_activity": datetime.now(),
                "annotation_count": 0,
                "is_active": True,
                "auto_save_enabled": True,
                "changes_since_save": 0,
            }

            with self._lock:
                self.active_sessions[session_id] = session_data

            self.logger.info(
                f"Labeling session created: {session_id} for {document_path}"
            )
            return session_id

        except Exception as e:
            self.logger.error(f"Failed to create labeling session: {str(e)}")
            raise LabelingError(
                message=f"Session creation failed: {str(e)}",
                session_id=None,
                original_exception=e,
            )

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        세션 정보 조회

        Args:
            session_id: 세션 ID

        Returns:
            Optional[Dict[str, Any]]: 세션 정보 (없으면 None)
        """
        with self._lock:
            return self.active_sessions.get(session_id)

    def update_session_activity(self, session_id: str) -> bool:
        """
        세션 활동 시간 업데이트

        Args:
            session_id: 세션 ID

        Returns:
            bool: 업데이트 성공 여부
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session["last_activity"] = datetime.now()
                return True
            return False

    def increment_annotation_count(self, session_id: str) -> bool:
        """
        세션의 어노테이션 카운트 증가

        Args:
            session_id: 세션 ID

        Returns:
            bool: 업데이트 성공 여부
        """
        with self._lock:
            session = self.active_sessions.get(session_id)
            if session:
                session["annotation_count"] += 1
                session["changes_since_save"] += 1
                session["last_activity"] = datetime.now()
                return True
            return False

    def close_session(self, session_id: str) -> bool:
        """
        세션 종료

        Args:
            session_id: 세션 ID

        Returns:
            bool: 종료 성공 여부
        """
        try:
            with self._lock:
                session = self.active_sessions.pop(session_id, None)
                if session:
                    session["is_active"] = False
                    session["closed_at"] = datetime.now()

                    self.logger.info(f"Labeling session closed: {session_id}")
                    return True
                return False

        except Exception as e:
            self.logger.error(f"Failed to close session {session_id}: {str(e)}")
            return False

    def get_session_statistics(self) -> Dict[str, Any]:
        """
        세션 통계 정보 반환

        Returns:
            Dict[str, Any]: 세션 통계
        """
        with self._lock:
            active_count = len(self.active_sessions)
            total_annotations = sum(
                session["annotation_count"] for session in self.active_sessions.values()
            )

            return {
                "active_sessions": active_count,
                "total_annotations": total_annotations,
                "max_sessions": ANNOTATION_MAX_ACTIVE_SESSIONS,
                "session_timeout_minutes": ANNOTATION_SESSION_TIMEOUT_MINUTES,
            }

    def _cleanup_expired_sessions(self) -> None:
        """
        만료된 세션 정리 (백그라운드 실행)
        """
        while True:
            try:
                current_time = datetime.now()
                expired_sessions = []

                with self._lock:
                    for session_id, session in self.active_sessions.items():
                        if (
                            current_time - session["last_activity"]
                            > self.session_timeout
                        ):
                            expired_sessions.append(session_id)

                for session_id in expired_sessions:
                    self.close_session(session_id)
                    self.logger.info(f"Expired session cleaned up: {session_id}")

                time.sleep(60)  # 1분마다 정리 실행

            except Exception as e:
                self.logger.error(f"Session cleanup failed: {str(e)}")
                time.sleep(60)


class LabelingService(BaseService, LabelingInterface):
    """
    라벨링 서비스 클래스

    문서 라벨링, 어노테이션 관리, 품질 제어 등의 라벨링 기능을 제공합니다.
    BaseService와 LabelingInterface를 구현합니다.
    """

    def __init__(self, config: ApplicationConfig, logger):
        """
        LabelingService 초기화

        Args:
            config: 애플리케이션 설정 객체
            logger: 로거 객체
        """
        super().__init__(config, logger)

        # 컴포넌트 초기화
        self.file_handler = FileHandler(config)
        self.annotation_manager = AnnotationManager(config)
        self.quality_controller = QualityController(config)
        self.session_manager = LabelingSessionManager(config)

        # 라벨링 템플릿 및 스키마
        self.annotation_templates: Dict[str, Dict[str, Any]] = {}
        self.validation_schemas: Dict[str, Dict[str, Any]] = {}

        # 상태 관리
        self.pending_documents: List[DocumentModel] = []
        self.in_progress_documents: List[DocumentModel] = []
        self.completed_documents: List[DocumentModel] = []

        # 통계 정보
        self.labeling_statistics: Dict[str, Any] = {}
        self.productivity_metrics: Dict[str, float] = {}

        # 진행 상태
        self.labeling_progress: Dict[str, float] = {}
        self.current_operation: Optional[str] = None
        self.processing_errors: List[str] = []

        # 콜백 관리
        self.progress_callbacks: List[Callable] = []
        self.completion_callbacks: List[Callable] = []

        # 스레드 안전성을 위한 락
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        """
        서비스 초기화

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info("Initializing LabelingService")

            # 상태 초기화
            with self._lock:
                self.pending_documents.clear()
                self.in_progress_documents.clear()
                self.completed_documents.clear()
                self.labeling_statistics.clear()
                self.productivity_metrics.clear()
                self.labeling_progress.clear()
                self.processing_errors.clear()
                self.current_operation = None

            # 어노테이션 템플릿 로드
            self._load_annotation_templates()

            # 검증 스키마 로드
            self._load_validation_schemas()

            self.logger.info("LabelingService initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize LabelingService: {str(e)}")
            self._is_initialized = False
            return False

    def cleanup(self) -> None:
        """
        서비스 정리
        """
        try:
            self.logger.info("Cleaning up LabelingService")

            # 모든 활성 세션 종료
            active_sessions = list(self.session_manager.active_sessions.keys())
            for session_id in active_sessions:
                self.session_manager.close_session(session_id)

            # 상태 정리
            with self._lock:
                self.pending_documents.clear()
                self.in_progress_documents.clear()
                self.completed_documents.clear()
                self.labeling_statistics.clear()
                self.productivity_metrics.clear()
                self.labeling_progress.clear()
                self.processing_errors.clear()

            self.logger.info("LabelingService cleanup completed")

        except Exception as e:
            self.logger.error(f"Error during LabelingService cleanup: {str(e)}")

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



    def create_labeling_session(self, file_path: str) -> str:
        """
        라벨링 세션 생성 (LabelingInterface 구현)

        Args:
            file_path: 파일 경로

        Returns:
            str: 세션 ID
        """
        try:
            self.logger.info(f"Creating labeling session for: {file_path}")

            # 파일 존재 확인
            if not os.path.exists(file_path):
                raise FileAccessError(
                    message=f"File not found: {file_path}",
                    file_path=file_path,
                    access_type="read",
                )

            # 세션 생성
            session_id = self.session_manager.create_session(file_path)

            # 문서 모델 생성
            document = DocumentModel.from_file_path(file_path)
            document.document_status = DocumentStatus.PROCESSING

            # 진행 중인 문서 목록에 추가
            with self._lock:
                self.in_progress_documents.append(document)

            # 진행 상태 업데이트
            self._update_labeling_progress()

            self.logger.info(f"Labeling session created: {session_id}")
            return session_id

        except Exception as e:
            self.logger.error(f"Failed to create labeling session: {str(e)}")
            raise LabelingError(
                message=f"Labeling session creation failed: {str(e)}",
                session_id=None,
                original_exception=e,
            )

    def get_labeling_progress(self) -> Dict[str, float]:
        """
        라벨링 진행 상황 제공 (LabelingInterface 구현)

        Returns:
            Dict[str, float]: 진행 상황
        """
        with self._lock:
            return self.labeling_progress.copy()

    def export_annotations(self, format_type: str) -> str:
        """
        어노테이션 내보내기 (LabelingInterface 구현)

        Args:
            format_type: 내보내기 형식

        Returns:
            str: 내보내기 결과
        """
        try:
            self.logger.info(f"Exporting annotations in format: {format_type}")

            # 완료된 문서들의 어노테이션 수집
            all_annotations = []
            with self._lock:
                for document in self.completed_documents:
                    if document.annotations:
                        all_annotations.extend(document.annotations)

            if not all_annotations:
                return "No annotations to export"

            # 형식별 내보내기
            if format_type.lower() == "json":
                return self._export_to_json(all_annotations)
            elif format_type.lower() == "xml":
                return self._export_to_xml(all_annotations)
            elif format_type.lower() == "csv":
                return self._export_to_csv(all_annotations)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

        except Exception as e:
            self.logger.error(f"Annotation export failed: {str(e)}")
            raise LabelingError(
                message=f"Annotation export failed: {str(e)}", original_exception=e
            )

    def save_annotation_data(
        self, session_id: str, annotation_data: Dict[str, Any]
    ) -> bool:
        """
        어노테이션 데이터 저장

        Args:
            session_id: 세션 ID
            annotation_data: 어노테이션 데이터

        Returns:
            bool: 저장 성공 여부
        """
        try:
            # 세션 유효성 확인
            session = self.session_manager.get_session(session_id)
            if not session:
                raise LabelingError(
                    message=f"Session not found: {session_id}", session_id=session_id
                )

            # 어노테이션 생성/업데이트
            document_id = annotation_data.get("document_id")
            page_number = annotation_data.get("page_number", 1)

            annotation = self.annotation_manager.create_annotation(
                document_id=document_id, page_number=page_number
            )

            # 필드 어노테이션 추가
            for field_data in annotation_data.get("fields", []):
                field_name = field_data["field_name"]
                field_type = AnnotationType(field_data["field_type"])

                # 바운딩 박스 생성
                bbox_data = field_data["bounding_box"]
                bounding_box = BoundingBox(
                    x=bbox_data["x"],
                    y=bbox_data["y"],
                    width=bbox_data["width"],
                    height=bbox_data["height"],
                )

                # 필드 어노테이션 추가
                self.annotation_manager.add_field_annotation(
                    annotation_id=annotation.annotation_id,
                    field_name=field_name,
                    field_type=field_type,
                    bounding_box=bounding_box,
                    text_value=field_data.get("text_value", ""),
                )

            # 세션 업데이트
            self.session_manager.increment_annotation_count(session_id)

            self.logger.info(f"Annotation data saved for session: {session_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save annotation data: {str(e)}")
            with self._lock:
                self.processing_errors.append(str(e))
            return False

    def load_annotation_template(self) -> Dict[str, Any]:
        """
        어노테이션 템플릿 로드

        Returns:
            Dict[str, Any]: 어노테이션 템플릿
        """
        try:
            template_path = os.path.join(
                self.config.templates_directory, "annotation_template.json"
            )

            if os.path.exists(template_path):
                with open(template_path, "r", encoding="utf-8") as f:
                    template = json.load(f)

                self.logger.info("Annotation template loaded successfully")
                return template
            else:
                # 기본 템플릿 반환
                return self._get_default_template()

        except Exception as e:
            self.logger.error(f"Failed to load annotation template: {str(e)}")
            return self._get_default_template()

    def validate_annotation_completeness(self, annotation_data: Dict[str, Any]) -> bool:
        """
        어노테이션 완성도 검증

        Args:
            annotation_data: 어노테이션 데이터

        Returns:
            bool: 완성도 검증 결과
        """
        try:
            # 필수 필드 확인
            required_fields = ["document_id", "page_number", "fields"]
            for field in required_fields:
                if field not in annotation_data:
                    return False

            # 필드 데이터 검증
            fields = annotation_data.get("fields", [])
            if not fields:
                return False

            # 각 필드의 완성도 검증
            for field in fields:
                if not self._validate_field_completeness(field):
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Annotation completeness validation failed: {str(e)}")
            return False

    def initialize_labeling_session(self, documents: List[DocumentModel]) -> bool:
        """
        라벨링 세션 초기화

        Args:
            documents: 문서 모델 목록

        Returns:
            bool: 초기화 성공 여부
        """
        try:
            self.logger.info(
                f"Initializing labeling session with {len(documents)} documents"
            )

            with self._lock:
                self.pending_documents.extend(documents)

            # 진행 상태 업데이트
            self._update_labeling_progress()

            self.logger.info("Labeling session initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize labeling session: {str(e)}")
            return False

    def get_labeling_statistics(self) -> Dict[str, Any]:
        """
        라벨링 통계 정보 반환

        Returns:
            Dict[str, Any]: 라벨링 통계
        """
        with self._lock:
            # 기본 통계 계산
            total_documents = (
                len(self.pending_documents)
                + len(self.in_progress_documents)
                + len(self.completed_documents)
            )

            annotation_stats = self.annotation_manager.get_statistics()
            quality_report = self.quality_controller.get_quality_report()
            session_stats = self.session_manager.get_session_statistics()

            return {
                "document_statistics": {
                    "total_documents": total_documents,
                    "pending_documents": len(self.pending_documents),
                    "in_progress_documents": len(self.in_progress_documents),
                    "completed_documents": len(self.completed_documents),
                },
                "annotation_statistics": annotation_stats,
                "quality_report": quality_report,
                "session_statistics": session_stats,
                "productivity_metrics": self.productivity_metrics,
                "processing_errors_count": len(self.processing_errors),
                "service_id": self.service_id,
            }

    def register_progress_callback(self, callback: Callable) -> None:
        """
        진행 상황 콜백 등록

        Args:
            callback: 콜백 함수
        """
        with self._lock:
            self.progress_callbacks.append(callback)

        self.logger.debug(f"Progress callback registered: {callback.__name__}")

    def register_completion_callback(self, callback: Callable) -> None:
        """
        완료 콜백 등록

        Args:
            callback: 콜백 함수
        """
        with self._lock:
            self.completion_callbacks.append(callback)

        self.logger.debug(f"Completion callback registered: {callback.__name__}")

    def _load_annotation_templates(self) -> None:
        """
        어노테이션 템플릿 로드
        """
        try:
            template_dir = Path(self.config.templates_directory)

            if template_dir.exists():
                for template_file in template_dir.glob("*.json"):
                    with open(template_file, "r", encoding="utf-8") as f:
                        template = json.load(f)
                        template_name = template_file.stem
                        self.annotation_templates[template_name] = template

                self.logger.info(
                    f"Loaded {len(self.annotation_templates)} annotation templates"
                )
            else:
                self.logger.warning("Templates directory not found")

        except Exception as e:
            self.logger.error(f"Failed to load annotation templates: {str(e)}")

    def _load_validation_schemas(self) -> None:
        """
        검증 스키마 로드
        """
        try:
            schema_path = os.path.join(
                self.config.templates_directory, "validation_schema.json"
            )

            if os.path.exists(schema_path):
                with open(schema_path, "r", encoding="utf-8") as f:
                    schema = json.load(f)
                    self.validation_schemas["default"] = schema

                self.logger.info("Validation schemas loaded successfully")
            else:
                self.logger.warning("Validation schema file not found")

        except Exception as e:
            self.logger.error(f"Failed to load validation schemas: {str(e)}")

    def _update_labeling_progress(self) -> None:
        """
        라벨링 진행 상황 업데이트
        """
        with self._lock:
            total_documents = (
                len(self.pending_documents)
                + len(self.in_progress_documents)
                + len(self.completed_documents)
            )

            if total_documents > 0:
                progress = {
                    "overall_progress": len(self.completed_documents) / total_documents,
                    "pending_ratio": len(self.pending_documents) / total_documents,
                    "in_progress_ratio": len(self.in_progress_documents)
                    / total_documents,
                    "completed_ratio": len(self.completed_documents) / total_documents,
                }
            else:
                progress = {
                    "overall_progress": 0.0,
                    "pending_ratio": 0.0,
                    "in_progress_ratio": 0.0,
                    "completed_ratio": 0.0,
                }

            self.labeling_progress = progress

        # 진행 상황 콜백 실행
        self._execute_progress_callbacks()

    def _execute_progress_callbacks(self) -> None:
        """
        진행 상황 콜백 실행
        """
        with self._lock:
            callbacks = self.progress_callbacks.copy()
            progress = self.labeling_progress.copy()

        for callback in callbacks:
            try:
                callback(progress)
            except Exception as e:
                self.logger.error(f"Progress callback execution failed: {str(e)}")

    def _execute_completion_callbacks(self) -> None:
        """
        완료 콜백 실행
        """
        with self._lock:
            callbacks = self.completion_callbacks.copy()
            completed_docs = self.completed_documents.copy()

        for callback in callbacks:
            try:
                callback(completed_docs)
            except Exception as e:
                self.logger.error(f"Completion callback execution failed: {str(e)}")

    def _get_default_template(self) -> Dict[str, Any]:
        """
        기본 어노테이션 템플릿 반환

        Returns:
            Dict[str, Any]: 기본 템플릿
        """
        return {
            "template_name": "default",
            "template_version": "1.0",
            "required_fields": [
                "document_title",
                "document_date",
                "supplier_name",
                "total_amount",
            ],
            "optional_fields": ["document_number", "supplier_address", "line_items"],
            "field_types": ANNOTATION_FIELD_TYPES,
        }

    def _validate_field_completeness(self, field_data: Dict[str, Any]) -> bool:
        """
        필드 완성도 검증

        Args:
            field_data: 필드 데이터

        Returns:
            bool: 완성도 검증 결과
        """
        required_keys = ["field_name", "field_type", "bounding_box", "text_value"]

        for key in required_keys:
            if key not in field_data:
                return False

            if key == "bounding_box":
                bbox = field_data[key]
                bbox_keys = ["x", "y", "width", "height"]
                if not all(k in bbox for k in bbox_keys):
                    return False

        return True

    def _export_to_json(self, annotations: List[AnnotationModel]) -> str:
        """
        JSON 형식으로 어노테이션 내보내기

        Args:
            annotations: 어노테이션 목록

        Returns:
            str: JSON 문자열
        """
        try:
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "annotation_count": len(annotations),
                "annotations": [annotation.to_dict() for annotation in annotations],
            }

            output_path = os.path.join(
                self.config.data_directory,
                "exports",
                f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)

            return output_path

        except Exception as e:
            self.logger.error(f"JSON export failed: {str(e)}")
            raise ProcessingError(f"JSON export failed: {str(e)}")

    def _export_to_xml(self, annotations: List[AnnotationModel]) -> str:
        """
        XML 형식으로 어노테이션 내보내기

        Args:
            annotations: 어노테이션 목록

        Returns:
            str: XML 파일 경로
        """
        try:
            # 간단한 XML 형식으로 내보내기
            output_path = os.path.join(
                self.config.data_directory,
                "exports",
                f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml",
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                f.write("<annotations>\n")

                for annotation in annotations:
                    f.write(f'  <annotation id="{annotation.annotation_id}">\n')
                    f.write(
                        f"    <document_id>{annotation.document_id}</document_id>\n"
                    )
                    f.write(
                        f"    <page_number>{annotation.page_number}</page_number>\n"
                    )
                    f.write("    <fields>\n")

                    for field in annotation.field_annotations.values():
                        f.write(
                            f'      <field name="{field.field_name}" type="{field.field_type.value}">\n'
                        )
                        f.write(
                            f"        <text_value>{field.text_value}</text_value>\n"
                        )
                        f.write(
                            f'        <bounding_box x="{field.bounding_box.x}" y="{field.bounding_box.y}" '
                        )
                        f.write(
                            f'width="{field.bounding_box.width}" height="{field.bounding_box.height}"/>\n'
                        )
                        f.write("      </field>\n")

                    f.write("    </fields>\n")
                    f.write("  </annotation>\n")

                f.write("</annotations>\n")

            return output_path

        except Exception as e:
            self.logger.error(f"XML export failed: {str(e)}")
            raise ProcessingError(f"XML export failed: {str(e)}")

    def _export_to_csv(self, annotations: List[AnnotationModel]) -> str:
        """
        CSV 형식으로 어노테이션 내보내기

        Args:
            annotations: 어노테이션 목록

        Returns:
            str: CSV 파일 경로
        """
        try:
            output_path = os.path.join(
                self.config.data_directory,
                "exports",
                f"annotations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8", newline="") as f:
                import csv

                writer = csv.writer(f)

                # 헤더 작성
                writer.writerow(
                    [
                        "annotation_id",
                        "document_id",
                        "page_number",
                        "field_name",
                        "field_type",
                        "text_value",
                        "bbox_x",
                        "bbox_y",
                        "bbox_width",
                        "bbox_height",
                        "confidence_score",
                        "is_validated",
                    ]
                )

                # 데이터 작성
                for annotation in annotations:
                    for field in annotation.field_annotations.values():
                        writer.writerow(
                            [
                                annotation.annotation_id,
                                annotation.document_id,
                                annotation.page_number,
                                field.field_name,
                                field.field_type.value,
                                field.text_value,
                                field.bounding_box.x,
                                field.bounding_box.y,
                                field.bounding_box.width,
                                field.bounding_box.height,
                                field.confidence_score,
                                field.is_validated,
                            ]
                        )

            return output_path

        except Exception as e:
            self.logger.error(f"CSV export failed: {str(e)}")
            raise ProcessingError(f"CSV export failed: {str(e)}")

    @classmethod
    def create_with_dependencies(cls, container) -> "LabelingService":
        """
        의존성 컨테이너를 사용한 팩토리 메서드

        Args:
            container: 의존성 컨테이너

        Returns:
            LabelingService: 생성된 서비스 인스턴스
        """
        return cls(
            config=container.get_service("config"),
            logger=container.get_service("logger"),
        )


# 모듈 수준 유틸리티 함수들
def create_labeling_service(config: ApplicationConfig) -> LabelingService:
    """
    라벨링 서비스 생성 함수

    Args:
        config: 애플리케이션 설정

    Returns:
        LabelingService: 생성된 서비스 인스턴스
    """
    logger = get_application_logger("labeling_service")
    service = LabelingService(config, logger)

    if not service.initialize():
        raise ProcessingError("Failed to initialize LabelingService")

    return service


if __name__ == "__main__":
    # 라벨링 서비스 테스트
    print("YOKOGAWA OCR 라벨링 서비스 테스트")
    print("=" * 50)

    try:
        # 설정 로드
        from config.settings import load_configuration

        config = load_configuration()

        # 서비스 생성
        service = create_labeling_service(config)

        # 상태 확인
        if service.health_check():
            print("✅ 라벨링 서비스 정상 동작")
        else:
            print("❌ 라벨링 서비스 상태 이상")

        # 통계 정보 출력
        statistics = service.get_labeling_statistics()
        print(f"📊 라벨링 통계: {statistics}")

        # 어노테이션 템플릿 로드 테스트
        template = service.load_annotation_template()
        print(f"📋 어노테이션 템플릿 로드: {template.get('template_name', 'Unknown')}")

        # 정리
        service.cleanup()

    except Exception as e:
        print(f"❌ 라벨링 서비스 테스트 실패: {e}")

    print("\n🎯 라벨링 서비스 구현이 완료되었습니다!")
