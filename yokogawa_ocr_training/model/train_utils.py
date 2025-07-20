#!/usr/bin/env python3
"""
YOKOGAWA OCR 인터랙티브 학습/검증 시스템 - 모델 학습/저장/예측 코어 유틸리티

지침: class-and-function.txt, data_flow_consistency_guidelines.txt,
project_architecture_guideline.txt, code_quality_validation_checklist.txt 엄수

작성자: YOKOGAWA OCR 개발팀
"""

import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path

import numpy as np
import tensorflow as tf

from models.document_model import DocumentModel
from models.annotation_model import AnnotationModel

CHECKPOINT_DIR = Path("model/model_checkpoint")

def load_training_dataset(annotations_path: str) -> List[Dict[str, Any]]:
    """
    어노테이션 데이터셋 로딩 함수

    Args:
        annotations_path (str): 어노테이션 JSON 파일 경로

    Returns:
        List[Dict[str, Any]]: 학습 데이터 리스트
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    return dataset

def build_model(config: Any) -> tf.keras.Model:
    """
    모델 생성 함수

    Args:
        config (Any): 모델 관련 설정(하이퍼파라미터 포함)

    Returns:
        tf.keras.Model: 신규 모델 객체
    """
    # 단순 CNN 기반 예시(실제 구조는 프로젝트 요구에 맞게 확장)
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, kernel_size=3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(config.get("num_classes", 10), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_one_epoch(model: tf.keras.Model, dataset: List[Dict[str, Any]], epoch: int) -> Dict[str, Any]:
    """
    한 epoch 단위 학습 실행

    Args:
        model (tf.keras.Model): 학습 대상 모델
        dataset (List[Dict[str, Any]]): 학습 데이터
        epoch (int): 현재 epoch(1-based)

    Returns:
        Dict[str, Any]: 학습 결과 메트릭
    """
    # 데이터 변환(이미지, 레이블 변환 등은 별도 모듈 사용)
    images, labels = _preprocess_training_data(dataset)
    history = model.fit(images, labels, epochs=1, shuffle=True, verbose=0)
    metrics = {k: v[-1] for k, v in history.history.items()}
    metrics["epoch"] = epoch
    return metrics

def save_checkpoint(model: tf.keras.Model, epoch: int, checkpoint_path: str) -> None:
    """
    체크포인트(h5) 저장

    Args:
        model (tf.keras.Model): 저장할 모델
        epoch (int): 저장 시점(epoch)
        checkpoint_path (str): 저장 경로

    Returns:
        None
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model.save(checkpoint_path)

def load_checkpoint(checkpoint_path: str) -> Tuple[tf.keras.Model, int]:
    """
    체크포인트(h5) 로딩

    Args:
        checkpoint_path (str): 파일 경로

    Returns:
        Tuple[tf.keras.Model, int]: 모델 객체 및 체크포인트 저장 epoch
    """
    model = tf.keras.models.load_model(checkpoint_path)
    # epoch 정보 파싱(파일명: "checkpoint_epoch_{epoch}.h5" 패턴)
    epoch = 0
    basename = os.path.basename(checkpoint_path)
    if "epoch_" in basename:
        try:
            epoch = int(basename.split("epoch_")[1].split(".")[0])
        except Exception:
            pass
    return model, epoch

def run_inference_on_validation(model: tf.keras.Model, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    검증셋에 대해 모델 추론(inference) 실행

    Args:
        model (tf.keras.Model): 추론 대상 모델
        dataset (List[Dict[str, Any]]): 검증 데이터

    Returns:
        List[Dict[str, Any]]: 예측 결과 리스트(json export용)
    """
    images, meta = _preprocess_validation_data(dataset)
    preds = model.predict(images, verbose=0)
    # 메타정보에 결과 결합
    inference_results = []
    for idx, pred in enumerate(preds):
        doc_id = meta[idx].get("document_id", f"val_{idx}")
        result = {
            "document_id": doc_id,
            "predict_proba": pred.tolist(),
            "file_path": meta[idx].get("file_path", ""),
            "status": "predicted",
            "annotations": _make_annotation_format_from_inference(pred, meta[idx])
        }
        inference_results.append(result)
    return inference_results

def _preprocess_training_data(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """
    학습 데이터셋 전처리(이미지, 라벨 추출 및 변환)

    Returns:
        Tuple[np.ndarray, np.ndarray]: 이미지 배열, 라벨(원-핫)
    """
    # 실제 구현은 데이터 구조에 맞게 보완
    image_list = []
    label_list = []
    for item in dataset:
        image_path = item.get("file_path")
        label = item.get("label", 0)
        image = _load_and_resize_image(image_path)
        image_list.append(image)
        label_list.append(label)
    images = np.stack(image_list)
    labels = tf.keras.utils.to_categorical(label_list)
    return images, labels

def _preprocess_validation_data(dataset: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    검증 데이터셋 전처리(이미지 및 메타정보 반환)
    """
    image_list = []
    meta_list = []
    for item in dataset:
        image_path = item.get("file_path")
        image = _load_and_resize_image(image_path)
        image_list.append(image)
        meta_list.append(item)
    return np.stack(image_list), meta_list

def _make_annotation_format_from_inference(pred: np.ndarray, meta_info: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    추론 결과로부터 annotation json 포맷 생성(실 구현은 도메인 요구 반영 확장)
    """
    label_index = int(np.argmax(pred))
    confidence = float(np.max(pred))
    annotation = {
        "annotation_id": f"{meta_info.get('document_id', 'unk')}_{label_index}",
        "document_id": meta_info.get("document_id", ""),
        "page_number": meta_info.get("page_number", 1),
        "annotation_type": "text",
        "annotation_status": "predicted",
        "bounding_boxes": [],
        "text_value": str(label_index),
        "confidence_score": confidence
    }
    return [annotation]

def _load_and_resize_image(image_path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    입력 이미지를 로딩하고 학습 입력 크기로 리사이즈

    Args:
        image_path (str): 이미지 경로
        size (Tuple[int, int]): 리사이즈 크기

    Returns:
        np.ndarray: 전처리된 이미지
    """
    from PIL import Image
    image = Image.open(image_path).convert("RGB").resize(size)
    arr = np.array(image) / 255.0
    return arr.astype(np.float32)
