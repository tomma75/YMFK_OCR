#!/usr/bin/env python3

"""
validation_interface/web_server.py

YOKOGAWA OCR 인터랙티브 학습/검증 시스템
- FastAPI 기반으로 inference 결과 export 및 사용자 피드백 수신을 위한 REST API 제공.
- 지침: class_interface_specifications.txt, variable_naming_standards.txt, code_quality_validation_checklist.txt 준수.

작성자: YOKOGAWA OCR 개발팀
"""

import os
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from models.annotation_model import AnnotationModel, BoundingBox  # 기존 선언된 모델 import
from utils.logger_util import get_application_logger  # 기존 선언된 로깅 유틸 import
from core.exceptions import ValidationError, FileProcessingError
from yokogawa_ocr_training.validation_interface.utils import save_json_file  # 기존 선언된 예외 import

# FastAPI 앱 생성
app = FastAPI(
    title="YOKOGAWA OCR Validation API",
    description="OCR 모델 예측 결과 조회 및 사용자 피드백 수신 API",
    version="1.0.0"
)
@app.get("/")
async def root():
    return {"status": "ok"}

# 로거 설정
logger = get_application_logger("web_server")

# 디렉터리 설정 (data_flow_consistency_guidelines.txt 준수)
INFERENCE_RESULTS_DIR = Path("validation_interface/inference_results")
FEEDBACK_SAVE_DIR = Path("validation_interface/feedback")
INFERENCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# DTO 클래스 정의 (class_interface_specifications.txt 기반: 인수/리턴 타입, DTO 구조 준수)
class BoundingBoxDTO(BaseModel):
    x: int = Field(..., ge=0, description="X 좌표")
    y: int = Field(..., ge=0, description="Y 좌표")
    width: int = Field(..., gt=0, description="너비")
    height: int = Field(..., gt=0, description="높이")
    page_number: int = Field(..., ge=1, description="페이지 번호")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수")

class AnnotationDTO(BaseModel):
    annotation_id: str = Field(..., description="어노테이션 ID")
    document_id: str = Field(..., description="문서 ID")
    page_number: int = Field(..., ge=1, description="페이지 번호")
    annotation_type: str = Field(..., description="어노테이션 타입")
    annotation_status: str = Field(..., description="어노테이션 상태")
    bounding_boxes: List[BoundingBoxDTO] = Field(..., description="바운딩 박스 목록")
    text_value: Optional[str] = Field(None, description="텍스트 값")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="신뢰도 점수")

class InferenceResultDTO(BaseModel):
    document_id: str = Field(..., description="문서 ID")
    file_path: str = Field(..., description="파일 경로")
    status: str = Field(..., description="상태")
    annotations: List[AnnotationDTO] = Field(..., description="어노테이션 목록")

class FeedbackRequestDTO(BaseModel):
    document_id: str = Field(..., description="문서 ID")
    user_id: str = Field(..., description="사용자 ID")
    reviewed_annotations: List[AnnotationDTO] = Field(..., description="검수된 어노테이션 목록")
    feedback_timestamp: Optional[str] = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="피드백 타임스탬프"
    )

# 엔드포인트 구현 (class_interface_specifications.txt: /get-inference-result, /review-feedback POST)
@app.get("/get-inference-result", response_model=List[InferenceResultDTO])
async def get_inference_result(document_id: Optional[str] = None) -> List[InferenceResultDTO]:
    """
    지정한 document_id의 예측 결과 리스트 반환 (document_id 미지정시 전체).
    - 입력: document_id (선택)
    - 출력: InferenceResultDTO 리스트
    - 에러: 404 (결과 없음), 500 (서버 오류)
    """
    try:
        results: List[InferenceResultDTO] = []
        for file in INFERENCE_RESULTS_DIR.glob("*.json"):
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if (document_id is None) or (data.get("document_id") == document_id):
                results.append(InferenceResultDTO(**data))
        if not results:
            raise HTTPException(status_code=404, detail="No inference results found")
        logger.info(f"Fetched {len(results)} inference results for document_id: {document_id}")
        return results
    except Exception as e:
        logger.error(f"Failed to fetch inference results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

@app.post("/review-feedback", status_code=201)
async def review_feedback(feedback: FeedbackRequestDTO) -> Dict[str, str]:
    """
    사용자 검수/수정 피드백 제출 및 저장.
    - 입력: FeedbackRequestDTO
    - 출력: success 메시지 및 feedback_id
    - 에러: 400 (잘못된 입력), 500 (저장 실패)
    """
    try:
        # 입력 데이터 검증 (Pydantic에 의해 자동 수행)
        unique_id = f"{feedback.document_id}_{feedback.user_id}_{uuid.uuid4().hex[:8]}"
        feedback_path = FEEDBACK_SAVE_DIR / f"{unique_id}.json"
        save_json_file(feedback.dict(), str(feedback_path))
        logger.info(f"Feedback saved: {unique_id}")
        return {"result": "success", "feedback_id": unique_id}
    except ValidationError as e:
        logger.warning(f"Invalid feedback data: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")

# 예외 핸들러 (code_quality_validation_checklist.txt: 에러/Interrupt 복구 설계)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    일반 예외 핸들러: 서버 오류 시 로그 기록 및 응답.
    """
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": str(exc)},
    )

# 서버 시작 로그
@app.on_event("startup")
async def startup_event():
    logger.info("Validation API server started")

# 서버 종료 로그 및 정리
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Validation API server shutting down")
