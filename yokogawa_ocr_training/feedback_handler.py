#!/usr/bin/env python3
"""
feedback_handler.py

YOKOGAWA OCR 인터랙티브 학습/검증 시스템
- 사용자 피드백(검증/수정 데이터)을 기존 어노테이션 파일과 원자적으로 병합하고, integrity를 보장하는 데이터셋 최신화 모듈.

참조: class-and-function.txt, variable_naming_standards.txt, data_flow_consistency_guidelines.txt, code_quality_validation_checklist.txt
"""

import os
import json
import shutil
import argparse
import logging
from typing import Any, Dict, List
from pathlib import Path
from datetime import datetime

from models.annotation_model import AnnotationModel
from core.exceptions import ValidationError, FileProcessingError
from validation_interface.utils import load_json_file, save_json_file

def merge_feedback_with_annotations(feedback_path: str, annotations_path: str) -> bool:
    """
    사용자 피드백 JSON을 기존 어노테이션 데이터와 병합하여 데이터셋을 최신화한다.

    Args:
        feedback_path (str): 검수자가 저장한 피드백(수정/검증 결과) JSON 경로
        annotations_path (str): 기존 어노테이션(JSON) 경로

    Returns:
        bool: 병합 성공 여부

    Raises:
        FileProcessingError: 파일 읽기 또는 쓰기 실패
        ValidationError: 데이터 형식 불일치, integrity 오류
    """
    # 1. 파일 로딩 및 백업
    annotations_backup_path = annotations_path + f".backup_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
    try:
        annotations_data = load_json_file(annotations_path)
        feedback_data = load_json_file(feedback_path)
        shutil.copy2(annotations_path, annotations_backup_path)
    except Exception as e:
        raise FileProcessingError(f"파일 로딩/백업 실패: {e}", file_path=annotations_path)

    # 2. 데이터 구조 검증 및 ID 기준 병합
    try:
        # List[Dict] 구조 가정 (annotation_id 기준)
        existing_annotations = {ann.get("annotation_id"): ann for ann in annotations_data}
        feedback_annotations = feedback_data.get("reviewed_annotations", [])
        updated = False
        for fb_ann in feedback_annotations:
            fb_id = fb_ann.get("annotation_id")
            if not fb_id:
                raise ValidationError("피드백에 annotation_id가 없습니다.")
            if fb_id in existing_annotations:
                # 피드백 내용으로 업데이트
                existing_annotations[fb_id] = fb_ann
                updated = True
            else:
                # 신규 어노테이션 추가(신규 라벨링 지원)
                existing_annotations[fb_id] = fb_ann
                updated = True
        if updated:
            merged_annotations = list(existing_annotations.values())
            # 3. 무결성 검사 (중복, 필드구조 등)
            annotation_ids = [a.get("annotation_id") for a in merged_annotations]
            if len(annotation_ids) != len(set(annotation_ids)):
                raise ValidationError("어노테이션 ID 중복 발생: 무결성 위반")
            # 4. 저장(atomic write)
            save_json_file(merged_annotations, annotations_path)
        else:
            # 병합할 변화가 없음
            return False
    except Exception as e:
        raise ValidationError(f"피드백 병합 오류: {e}")

    return True

def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="피드백/검증 결과 어노테이션 병합 툴",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--feedback-path", type=str, required=True, help="피드백 파일(JSON) 경로 또는 디렉터리")
    parser.add_argument("--annotations-path", type=str, required=True, help="기존 어노테이션(JSON) 경로")
    return parser.parse_args()

def main() -> None:
    args = _parse_cli_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # 지원: --feedback-path 디렉터리 병합(bulk)
    feedback_target = args.feedback_path
    annotation_target = args.annotations_path
    if os.path.isdir(feedback_target):
        feedback_files = [str(p) for p in Path(feedback_target).glob("*.json")]
        for feedback_file in feedback_files:
            try:
                result = merge_feedback_with_annotations(feedback_file, annotation_target)
                if result:
                    logging.info(f"병합 성공: {feedback_file}")
                else:
                    logging.info(f"병합 내용 없음: {feedback_file}")
            except Exception as e:
                logging.error(f"병합 실패: {feedback_file} → {e}")
    else:
        try:
            result = merge_feedback_with_annotations(feedback_target, annotation_target)
            if result:
                logging.info(f"병합 성공: {feedback_target}")
            else:
                logging.info(f"병합 내용 없음: {feedback_target}")
        except Exception as e:
            logging.error(f"병합 실패: {feedback_target} → {e}")

if __name__ == "__main__":
    main()
