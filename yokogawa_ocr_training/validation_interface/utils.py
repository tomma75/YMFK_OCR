"""
validation_interface/utils.py

YOKOGAWA OCR 인터랙티브 학습/검증 시스템
검증, 파일 입출력, 포맷 변환, 데이터 일관성 체크 등 UI/백엔드/피드백 통합 지원 유틸리티

참조: class_interface_specifications.txt, data_flow_consistency_guidelines.txt,
variable_naming_standards.txt, code_quality_validation_checklist.txt

작성자: YOKOGAWA OCR 개발팀
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

from models.annotation_model import AnnotationModel, BoundingBox, AnnotationStatus
from models.document_model import DocumentModel
from core.exceptions import ValidationError, FileProcessingError

# ==============================
# 1. 파일 I/O 및 경로 유틸리티
# ==============================

def load_json_file(json_path: str) -> Any:
    """
    JSON 파일을 로딩합니다.

    Args:
        json_path (str): 파일 경로

    Returns:
        Any: 로드된 Python 객체

    Raises:
        FileProcessingError: 파일 읽기 실패 시
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise FileProcessingError(f"Failed to read JSON file: {e}", file_path=json_path)

def save_json_file(data: Any, json_path: str, indent: int = 2) -> None:
    """
    데이터를 JSON 파일로 저장합니다.

    Args:
        data (Any): 저장할 데이터
        json_path (str): 파일 경로
        indent (int): 들여쓰기 수준

    Raises:
        FileProcessingError: 파일 쓰기 실패 시
    """
    try:
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
    except Exception as e:
        raise FileProcessingError(f"Failed to write JSON file: {e}", file_path=json_path)

def list_json_files(directory: str) -> List[str]:
    """
    지정 디렉터리의 모든 JSON 파일 목록을 반환합니다.

    Args:
        directory (str): 검색할 디렉터리

    Returns:
        List[str]: JSON 파일 경로 목록
    """
    return [str(f) for f in Path(directory).glob("*.json")]

# ===================================
# 2. 어노테이션 및 예측 결과 변환 유틸
# ===================================

def annotationmodel_to_ui_dict(annotation: AnnotationModel) -> Dict[str, Any]:
    """
    AnnotationModel 객체를 UI에서 사용하는 dict 포맷으로 변환

    Args:
        annotation (AnnotationModel): 어노테이션 모델 객체

    Returns:
        Dict[str, Any]: UI 전송용 dict
    """
    return annotation.to_dict()

def parse_feedback_from_ui(feedback_data: Dict[str, Any]) -> AnnotationModel:
    """
    사용자 피드백 JSON(dict)에서 AnnotationModel로 변환

    Args:
        feedback_data (Dict[str, Any]): UI 피드백 데이터

    Returns:
        AnnotationModel: 파싱된 모델 객체

    Raises:
        ValidationError: 데이터 구조 불일치/유효성 실패 시
    """
    try:
        return AnnotationModel.from_dict(feedback_data)
    except Exception as e:
        raise ValidationError(f"Failed to parse feedback data: {e}")

def validate_annotation_format(annotation_data: Dict[str, Any]) -> bool:
    """
    어노테이션 dict 포맷 유효성 검사 (UI->API 데이터 sanity check)

    Args:
        annotation_data (Dict[str, Any]): 어노테이션 dict

    Returns:
        bool: 유효하면 True, 아니면 False
    """
    required = {"annotation_id", "document_id", "page_number", "annotation_type"}
    return required.issubset(annotation_data)

# ========================================
# 3. 시각화 지원/어노테이션 통계 유틸리티
# ========================================

def bounding_box_to_tuple(bbox: BoundingBox) -> Tuple[int, int, int, int]:
    """
    BoundingBox 객체를 (x, y, width, height) 튜플로 변환

    Args:
        bbox (BoundingBox): 바운딩 박스 객체

    Returns:
        Tuple[int, int, int, int]:
    """
    return (bbox.x, bbox.y, bbox.width, bbox.height)

def aggregate_annotation_statistics(annotations: List[AnnotationModel]) -> Dict[str, Any]:
    """
    어노테이션 목록에 대한 통계치 계산 (UI 통계, 검증대시보드용)

    Args:
        annotations (List[AnnotationModel]): 어노테이션 리스트

    Returns:
        Dict[str, Any]: 통계 정보
    """
    total = len(annotations)
    status_counts = {}
    for ann in annotations:
        status = getattr(ann, "annotation_status", AnnotationStatus.PENDING)
        status_counts[status] = status_counts.get(status, 0) + 1
    return {
        "total_annotations": total,
        "status_counts": {s.value if hasattr(s, 'value') else s: c for s, c in status_counts.items()},
    }

# ================================
# 4. 데이터 무결성/일관성 체크 함수
# ================================

def check_duplicate_annotations(annotations: List[AnnotationModel]) -> bool:
    """
    어노테이션 ID 기준 중복 여부 체크

    Args:
        annotations (List[AnnotationModel]): 어노테이션 리스트

    Returns:
        bool: 중복 없으면 True
    """
    ids = [ann.annotation_id for ann in annotations]
    return len(ids) == len(set(ids))

def check_bounding_box_valid(bbox: BoundingBox) -> bool:
    """
    바운딩 박스 좌표 및 크기 유효성 검증

    Args:
        bbox (BoundingBox): 바운딩 박스

    Returns:
        bool: 유효성 여부
    """
    return bbox.width > 0 and bbox.height > 0 and bbox.x >= 0 and bbox.y >= 0

# ============================
# 5. 일반 보조 유틸리티 함수
# ============================

def get_current_timestamp_str() -> str:
    """
    ISO 포맷 현재 타임스탬프 문자열 반환

    Returns:
        str
    """
    return datetime.utcnow().isoformat()

def safe_mkdir(path: str) -> None:
    """
    디렉터리를 안전하게 생성

    Args:
        path (str): 경로
    """
    os.makedirs(path, exist_ok=True)
