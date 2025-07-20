#!/usr/bin/env python3
"""
YOKOGAWA OCR 인터랙티브 학습/검증 시스템 - 메인 오케스트레이터

학습, 검증, 피드백 병합, 중간 체크포인트 관리 및 UI와의 실시간 연동을 담당하는 엔트리포인트입니다.

지침서: class-and-function.txt, data_flow_consistency_guidelines.txt, project_architecture_guideline.txt, code_quality_validation_checklist.txt를 엄수하여 작성.

작성자: YOKOGAWA OCR 개발팀
"""

import os
import sys
import time
import signal
import argparse
import logging
import threading
import requests
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# 프로젝트 내 실제 선언된 모듈 및 클래스만 import
from config.settings import load_configuration, ApplicationConfig
from utils.logger_util import setup_logger
from model.train_utils import (
    load_training_dataset,
    build_model,
    train_one_epoch,
    save_checkpoint,
    run_inference_on_validation,
    load_checkpoint,
)
from feedback_handler import merge_feedback_with_annotations
from validation_interface.web_server import app

CHECKPOINT_DIR = Path("model/model_checkpoint")
INFERENCE_OUTPUT_DIR = Path("validation_interface/inference_results")
FEEDBACK_INPUT_DIR = Path("validation_interface/feedback")
ANNOTATION_DATASET_PATH = Path("data/annotations/annotations.json")
LOGS_DIR = Path("logs")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="[YOKOGAWA OCR] 인터랙티브 학습/검증 오케스트레이터",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=10, help="총 학습 epoch")
    parser.add_argument("--resume", action="store_true", help="이전 체크포인트에서 이어서 학습")
    parser.add_argument("--stop-after-feedback", action="store_true", help="피드백 반영 후 자동 일시정지")
    parser.add_argument("--checkpoint-every", type=int, default=1, help="몇 epoch마다 체크포인트 저장")
    parser.add_argument("--validate-every", type=int, default=1, help="몇 epoch마다 검증 및 inference export")
    return parser.parse_args()

class TrainingManager:
    def __init__(self, config: ApplicationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.current_epoch = 0
        self.stop_training = False
        self.model = None
        self.dataset = None

    def load_or_init_model(self, resume: bool):
        if resume and any(CHECKPOINT_DIR.glob("*.h5")):
            latest_ckpt = sorted(CHECKPOINT_DIR.glob("*.h5"))[-1]
            self.model, self.current_epoch = load_checkpoint(str(latest_ckpt))
            self.logger.info(f"체크포인트에서 모델 복원: {latest_ckpt} (epoch={self.current_epoch})")
        else:
            self.model = build_model(self.config)
            self.current_epoch = 0
            self.logger.info("신규 모델 생성 및 학습 시작")

    def run_training(self, args: argparse.Namespace):
        self.dataset = load_training_dataset(str(ANNOTATION_DATASET_PATH))
        for epoch in range(self.current_epoch + 1, args.epochs + 1):
            if self.stop_training:
                self.logger.info(f"학습 중단 요청 감지. epoch {epoch}에서 종료.")
                break
            self.logger.info(f"[Epoch {epoch}] 학습 시작")
            train_stats = train_one_epoch(self.model, self.dataset, epoch)
            self.logger.info(f"[Epoch {epoch}] 학습 결과: {train_stats}")

            if epoch % args.checkpoint_every == 0:
                ckpt_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.h5"
                save_checkpoint(self.model, epoch, str(ckpt_path))
                self.logger.info(f"체크포인트 저장됨: {ckpt_path}")

            if epoch % args.validate_every == 0:
                self.logger.info(f"[Epoch {epoch}] 검증셋 inference 실행 및 UI export")
                inference_results = run_inference_on_validation(self.model, self.dataset)
                self._export_inference_to_api(inference_results)

                if self._wait_for_feedback_and_handle():
                    if args.stop_after_feedback:
                        self.logger.info("피드백 1회 반영 후 학습 일시정지")
                        break
        self.logger.info("모든 학습 종료/중단됨")

    def _export_inference_to_api(self, inference_results: List[Dict[str, Any]]):
        try:
            INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            for doc_result in inference_results:
                file_name = f"{doc_result['document_id']}_inference.json"
                file_path = INFERENCE_OUTPUT_DIR / file_name
                with open(file_path, "w", encoding="utf-8") as f:
                    import json
                    json.dump(doc_result, f, ensure_ascii=False, indent=2)
            self.logger.info(f"inference 결과 {len(inference_results)}건 export 완료")
            # 웹 API 연동 (선택) 예시:
            requests.post("http://localhost:8000/upload-inference", json=inference_results)
        except Exception as e:
            self.logger.warning(f"inference export 실패: {e}")

    def _wait_for_feedback_and_handle(self) -> bool:
        """피드백 폴더에 새 피드백 json이 생길 때까지 polling; 발견 시 병합후 True 반환(없으면 False)"""
        FEEDBACK_INPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.logger.info("피드백 대기 중...(웹 UI에서 검토 후 제출 필요)")
        poll_interval = 5
        max_wait = 60 * 30  # 30분 타임아웃
        waited = 0
        while waited < max_wait and not self.stop_training:
            feedback_files = list(FEEDBACK_INPUT_DIR.glob("*.json"))
            if feedback_files:
                for feedback_file in feedback_files:
                    try:
                        merge_feedback_with_annotations(
                            feedback_path=str(feedback_file),
                            annotations_path=str(ANNOTATION_DATASET_PATH)
                        )
                        self.logger.info(f"피드백 병합 및 업데이트 성공: {feedback_file.name}")
                        feedback_file.unlink()  # 병합 후 삭제/아카이브
                    except Exception as e:
                        self.logger.error(f"피드백 병합 오류: {feedback_file.name} : {e}")
                return True  # 담당 feedback 처리 완료
            time.sleep(poll_interval)
            waited += poll_interval
        self.logger.info("피드백 입력 없음(타임아웃/skip)")
        return False

def _signal_handler(trainer: TrainingManager, signum, frame):
    print("\n신호 감지됨(중단): 안전하게 학습 중단 절차 진입")
    trainer.stop_training = True

def main():
    # config 로딩
    config = load_configuration()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("interactive_trainer", config.logging_config)
    args = parse_arguments()

    # 학습 제어자 생성 및 신호 연동
    trainer = TrainingManager(config, logger)
    signal.signal(signal.SIGINT, lambda s, f: _signal_handler(trainer, s, f))
    signal.signal(signal.SIGTERM, lambda s, f: _signal_handler(trainer, s, f))

    try:
        trainer.load_or_init_model(args.resume)
        trainer.run_training(args)
    except Exception as e:
        logger.error(f"학습 오케스트레이터 예외: {e}", exc_info=1)
    finally:
        logger.info("interactive_trainer 종료")

if __name__ == "__main__":
    main()
