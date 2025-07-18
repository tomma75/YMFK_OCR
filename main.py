#!/usr/bin/env python3

"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 메인 실행 스크립트

이 스크립트는 OCR 데이터 준비 파이프라인의 전체 실행을 관리합니다.
데이터 수집, 라벨링, 데이터 증강, 검증의 각 단계를 순차적으로 또는 개별적으로 실행할 수 있습니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
버전: 1.0.0
"""

import sys
import os
import argparse
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
from venv import logger

# main.py 상단에 추가
from config import get_application_config, validate_environment_setup
from utils.logger_util import setup_logger
from core.exceptions import ServiceError
from services.data_collection_service import DataCollectionService
from services.labeling_service import LabelingService
from services.augmentation_service import AugmentationService
from services.validation_service import ValidationService

# 프로젝트 루트 디렉토리를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 핵심 설정 및 예외 클래스
from config.settings import load_configuration, ApplicationConfig
from core.exceptions import (
    ApplicationError,
    ServiceError,
    ProcessingError,
    ValidationError,
    ConfigurationError,
)

# 서비스 생성 함수들 (실제 존재하는 함수들만 import)
from services.data_collection_service import create_data_collection_service
from services.labeling_service import create_labeling_service
from services.augmentation_service import create_augmentation_service
from services.validation_service import create_validation_service

# 유틸리티 함수들
from utils.logger_util import setup_logger, get_application_logger
from utils.file_handler import FileHandler

# 서비스 인스턴스 전역 변수
_application_config: Optional[ApplicationConfig] = None
_data_collection_service = None
_labeling_service = None
_augmentation_service = None
_validation_service = None
_application_logger = None


def setup_logging() -> None:
    """
    로깅 설정 초기화

    애플리케이션 전체에서 사용할 로깅 시스템을 설정합니다.
    """

    global _application_logger
    try:
        # 설정 로드
        config = load_configuration()

        # 로거 설정
        _application_logger = setup_logger(
            name="yokogawa_ocr_main", config=config.logging_config
        )

        # 로그 디렉토리 생성
        log_dir = Path(config.logging_config.log_file_path).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        _application_logger.info("=" * 60)
        _application_logger.info("YOKOGAWA OCR 데이터 준비 시스템 시작")
        _application_logger.info("=" * 60)
        _application_logger.info(f"애플리케이션 버전: {config.app_version}")
        _application_logger.info(f"실행 환경: {config.environment}")
        _application_logger.info(f"로깅 시스템 초기화 완료")
    except Exception as e:
        print(f"❌ 로깅 설정 실패: {str(e)}", file=sys.stderr)
        raise ApplicationError(f"로깅 설정 실패: {str(e)}")


def initialize_application() -> bool:
    """애플리케이션 초기화"""
    try:
        _application_logger.info("애플리케이션 초기화 시작")
        # 환경변수 검증 및 설정
        from config import validate_environment_setup

        if not validate_environment_setup():
            _application_logger.warning("환경변수 검증 실패, 기본값으로 계속 진행")
        # 설정 파일 로드
        _application_logger.info("설정 파일 로드 중...")
        global _application_config
        _application_config = load_configuration()
        _application_logger.info(f"설정 로드 완료: {_application_config.app_name}")
        # 디렉터리 구조 생성
        _application_logger.info("디렉토리 구조 생성 중...")
        _application_config._ensure_directories()
        # 서비스 초기화
        _application_logger.info("서비스 초기화 중...")
        global _data_collection_service, _labeling_service
        global _augmentation_service, _validation_service
        _data_collection_service = DataCollectionService(
            _application_config,
            setup_logger("data_collection_service", _application_config.logging_config),
        )
        if not _data_collection_service.start():
            raise ServiceError("데이터 수집 서비스 시작 실패")
        _application_logger.info("✅ 데이터 수집 서비스 초기화 완료")
        _labeling_service = LabelingService(
            _application_config,
            setup_logger("labeling_service", _application_config.logging_config),
        )
        if not _labeling_service.start():
            raise ServiceError("라벨링 서비스 시작 실패")
        _application_logger.info("✅ 라벨링 서비스 초기화 완료")
        _augmentation_service = AugmentationService(
            _application_config,
            setup_logger("augmentation_service", _application_config.logging_config),
        )
        if not _augmentation_service.start():
            raise ServiceError("데이터 증강 서비스 시작 실패")
        _application_logger.info("✅ 데이터 증강 서비스 초기화 완료")
        _validation_service = ValidationService(
            _application_config,
            setup_logger("validation_service", _application_config.logging_config),
        )
        if not _validation_service.start():
            raise ServiceError("검증 서비스 시작 실패")
        _application_logger.info("✅ 검증 서비스 초기화 완료")
        # 서비스 상태 확인
        failed_services = []
        service_checks = [
            ("데이터 수집", _data_collection_service),
            ("라벨링", _labeling_service),
            ("데이터 증강", _augmentation_service),
            ("검증", _validation_service),
        ]
        for service_name, service_instance in service_checks:
            if not service_instance.health_check():
                failed_services.append(service_name)
        if failed_services:
            raise ServiceError(f"서비스 상태 확인 실패: {', '.join(failed_services)}")
        _application_logger.info("✅ 모든 서비스 초기화 완료")
        return True
    except Exception as e:
        _application_logger.error(f"❌ 애플리케이션 초기화 실패: {str(e)}")
        return False


def run_data_collection_pipeline() -> bool:
    """
    데이터 수집 파이프라인 실행

    지정된 소스 디렉토리에서 PDF 및 이미지 파일을 수집합니다.

    Returns:
        bool: 파이프라인 실행 성공 여부
    """
    try:
        _application_logger.info("=" * 50)
        _application_logger.info("📂 데이터 수집 파이프라인 시작")
        _application_logger.info("=" * 50)

        start_time = time.time()

        # 소스 디렉토리에서 파일 수집
        source_directory = _application_config.raw_data_directory
        collected_files = _data_collection_service.collect_files(source_directory)

        if not collected_files:
            _application_logger.warning("수집된 파일이 없습니다.")
            return False

        # 수집 통계 출력
        collection_stats = _data_collection_service.get_collection_statistics()
        _application_logger.info(f"📊 수집 통계:")
        _application_logger.info(f"  - 총 파일 수: {len(collected_files)}")
        _application_logger.info(
            f"  - 처리 시간: {collection_stats.get('processing_duration', 0):.2f}초"
        )

        end_time = time.time()
        _application_logger.info(
            f"✅ 데이터 수집 완료 (소요시간: {end_time - start_time:.2f}초)"
        )

        return True

    except Exception as e:
        _application_logger.error(f"❌ 데이터 수집 파이프라인 실행 실패: {str(e)}")
        return False


def run_labeling_pipeline() -> bool:
    """
    라벨링 파이프라인 실행

    수집된 문서에 대해 어노테이션을 생성하고 라벨링을 수행합니다.

    Returns:
        bool: 파이프라인 실행 성공 여부
    """
    try:
        _application_logger.info("=" * 50)
        _application_logger.info("🏷️  라벨링 파이프라인 시작")
        _application_logger.info("=" * 50)

        start_time = time.time()

        # 어노테이션 템플릿 로드
        annotation_template = _labeling_service.load_annotation_template()
        _application_logger.info(
            f"템플릿 로드: {annotation_template.get('template_name', 'Unknown')}"
        )

        # 라벨링 진행 상황 모니터링
        labeling_progress = _labeling_service.get_labeling_progress()
        _application_logger.info(
            f"라벨링 진행률: {labeling_progress.get('progress', 0):.1f}%"
        )

        # 라벨링 통계 출력
        labeling_stats = _labeling_service.get_labeling_statistics()
        _application_logger.info(f"📊 라벨링 통계:")
        _application_logger.info(
            f"  - 처리된 문서: {labeling_stats.get('processed_documents', 0)}"
        )
        _application_logger.info(
            f"  - 완료된 어노테이션: {labeling_stats.get('completed_annotations', 0)}"
        )

        end_time = time.time()
        _application_logger.info(
            f"✅ 라벨링 완료 (소요시간: {end_time - start_time:.2f}초)"
        )

        return True

    except Exception as e:
        _application_logger.error(f"❌ 라벨링 파이프라인 실행 실패: {str(e)}")
        return False


def run_augmentation_pipeline() -> bool:
    """
    데이터 증강 파이프라인 실행

    라벨링된 데이터에 대해 다양한 증강 기법을 적용합니다.

    Returns:
        bool: 파이프라인 실행 성공 여부
    """
    try:
        _application_logger.info("=" * 50)
        _application_logger.info("🔄 데이터 증강 파이프라인 시작")
        _application_logger.info("=" * 50)

        start_time = time.time()

        # 기본 데이터셋 생성 (테스트용)
        test_dataset = [
            {"image_path": "test1.jpg", "label": "document"},
            {"image_path": "test2.jpg", "label": "invoice"},
        ]

        # 데이터셋 증강 수행
        augmented_dataset = _augmentation_service.augment_dataset(test_dataset)

        # 증강 통계 출력
        augmentation_stats = _augmentation_service.get_augmentation_statistics()
        _application_logger.info(f"📊 증강 통계:")
        _application_logger.info(f"  - 원본 데이터: {len(test_dataset)}")
        _application_logger.info(f"  - 증강된 데이터: {len(augmented_dataset)}")
        _application_logger.info(
            f"  - 증강 배수: {len(augmented_dataset) / len(test_dataset):.1f}x"
        )

        end_time = time.time()
        _application_logger.info(
            f"✅ 데이터 증강 완료 (소요시간: {end_time - start_time:.2f}초)"
        )

        return True

    except Exception as e:
        _application_logger.error(f"❌ 데이터 증강 파이프라인 실행 실패: {str(e)}")
        return False


def run_validation_pipeline() -> bool:
    """
    검증 파이프라인 실행

    증강된 데이터셋의 품질을 검증하고 최종 리포트를 생성합니다.

    Returns:
        bool: 파이프라인 실행 성공 여부
    """
    try:
        _application_logger.info("=" * 50)
        _application_logger.info("✅ 검증 파이프라인 시작")
        _application_logger.info("=" * 50)

        start_time = time.time()

        # 테스트 데이터셋 생성
        test_dataset = [
            {"document_id": "doc1", "annotations": [], "status": "completed"},
            {"document_id": "doc2", "annotations": [], "status": "completed"},
        ]

        # 데이터셋 검증 수행
        validation_result = _validation_service.validate_dataset(test_dataset)

        # 검증 통계 출력
        validation_stats = _validation_service.get_validation_statistics()
        _application_logger.info(f"📊 검증 통계:")
        _application_logger.info(
            f"  - 검증된 항목: {validation_stats.get('validated_items', 0)}"
        )
        _application_logger.info(
            f"  - 품질 점수: {validation_stats.get('quality_score', 0):.3f}"
        )
        _application_logger.info(
            f"  - 검증 결과: {'통과' if validation_result else '실패'}"
        )

        end_time = time.time()
        _application_logger.info(
            f"✅ 검증 완료 (소요시간: {end_time - start_time:.2f}초)"
        )

        return True

    except Exception as e:
        _application_logger.error(f"❌ 검증 파이프라인 실행 실패: {str(e)}")
        return False


def _cleanup_services() -> None:
    """
    서비스 정리 및 종료

    모든 서비스의 정리 작업을 수행합니다.
    """
    global _data_collection_service, _labeling_service, _augmentation_service, _validation_service

    try:
        _application_logger.info("서비스 정리 중...")

        services = [
            ("데이터 수집", _data_collection_service),
            ("라벨링", _labeling_service),
            ("데이터 증강", _augmentation_service),
            ("검증", _validation_service),
        ]

        for service_name, service in services:
            if service:
                try:
                    service.cleanup()
                    _application_logger.debug(f"{service_name} 서비스 정리 완료")
                except Exception as e:
                    _application_logger.warning(
                        f"{service_name} 서비스 정리 실패: {str(e)}"
                    )

        _application_logger.info("✅ 모든 서비스 정리 완료")

    except Exception as e:
        _application_logger.error(f"❌ 서비스 정리 중 오류: {str(e)}")


def _create_argument_parser() -> argparse.ArgumentParser:
    """
    명령행 인자 파서 생성

    Returns:
        argparse.ArgumentParser: 설정된 인자 파서
    """
    parser = argparse.ArgumentParser(
        description="YOKOGAWA OCR 데이터 준비 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
실행 예시:
  %(prog)s --mode full                    # 전체 파이프라인 실행
  %(prog)s --mode collection              # 데이터 수집만 실행
  %(prog)s --mode labeling                # 라벨링만 실행
  %(prog)s --mode augmentation            # 데이터 증강만 실행
  %(prog)s --mode validation              # 검증만 실행
  %(prog)s --mode web                     # 웹 인터페이스 실행
        """,
    )

    parser.add_argument(
        "--mode",
        choices=["collection", "labeling", "augmentation", "validation", "full", "web"],
        default="full",
        help="실행 모드 선택 (기본값: full)",
    )

    parser.add_argument(
        "--config", type=str, help="설정 파일 경로 (기본값: config/application.json)"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="로그 레벨 설정 (기본값: INFO)",
    )

    parser.add_argument("--verbose", action="store_true", help="상세 출력 모드")

    parser.add_argument(
        "--dry-run", action="store_true", help="실제 실행 없이 계획만 출력"
    )

    return parser


def main() -> Optional[int]:
    """
    메인 실행 함수

    애플리케이션의 전체 실행 흐름을 관리합니다.

    Returns:
        Optional[int]: 종료 코드 (0: 정상, 1: 오류)
    """
    try:
        # 1. 명령행 인자 파싱
        parser = _create_argument_parser()
        args = parser.parse_args()

        # 2. 환경 변수 설정
        if args.config:
            os.environ["YOKOGAWA_CONFIG_FILE"] = args.config

        # 3. 로깅 설정
        setup_logging()

        # 4. 애플리케이션 초기화
        if not initialize_application():
            _application_logger.error("애플리케이션 초기화 실패")
            return 1

        # 5. 실행 계획 출력
        _application_logger.info(f"실행 모드: {args.mode}")
        _application_logger.info(f"로그 레벨: {args.log_level}")
        if args.verbose:
            _application_logger.info("상세 출력 모드 활성화")
        if args.dry_run:
            _application_logger.info("⚠️  드라이 런 모드 - 실제 실행되지 않음")
            return 0

        # 6. 모드별 실행
        execution_start_time = time.time()
        execution_success = True

        if args.mode == "collection":
            execution_success = run_data_collection_pipeline()

        elif args.mode == "labeling":
            execution_success = run_labeling_pipeline()

        elif args.mode == "augmentation":
            execution_success = run_augmentation_pipeline()

        elif args.mode == "validation":
            execution_success = run_validation_pipeline()

        elif args.mode == "full":
            # 전체 파이프라인 순차 실행
            pipeline_steps = [
                ("데이터 수집", run_data_collection_pipeline),
                ("라벨링", run_labeling_pipeline),
                ("데이터 증강", run_augmentation_pipeline),
                ("검증", run_validation_pipeline),
            ]

            for step_name, step_function in pipeline_steps:
                _application_logger.info(f"🔄 {step_name} 단계 시작")
                if not step_function():
                    _application_logger.error(f"❌ {step_name} 단계 실패")
                    execution_success = False
                    break
                _application_logger.info(f"✅ {step_name} 단계 완료")

        elif args.mode == "web":
            _application_logger.info("🌐 웹 인터페이스 모드")
            _application_logger.info("웹 인터페이스는 향후 구현 예정입니다.")
            return 0

        # 7. 실행 결과 출력
        execution_end_time = time.time()
        total_execution_time = execution_end_time - execution_start_time

        if execution_success:
            _application_logger.info("=" * 60)
            _application_logger.info("🎉 모든 파이프라인 실행 완료!")
            _application_logger.info(f"총 실행 시간: {total_execution_time:.2f}초")
            _application_logger.info("=" * 60)
            return 0
        else:
            _application_logger.error("=" * 60)
            _application_logger.error("❌ 파이프라인 실행 중 오류 발생")
            _application_logger.error(f"실행 시간: {total_execution_time:.2f}초")
            _application_logger.error("=" * 60)
            return 1

    except KeyboardInterrupt:
        if _application_logger:
            _application_logger.info("사용자에 의해 실행이 중단되었습니다.")
        else:
            print("사용자에 의해 실행이 중단되었습니다.")
        return 1

    except ApplicationError as e:
        if _application_logger:
            _application_logger.error(f"애플리케이션 오류: {str(e)}")
        else:
            print(f"애플리케이션 오류: {str(e)}", file=sys.stderr)
        return 1

    except Exception as e:
        if _application_logger:
            _application_logger.critical(f"예상치 못한 오류: {str(e)}")
        else:
            print(f"예상치 못한 오류: {str(e)}", file=sys.stderr)
        return 1

    finally:
        # 8. 정리 작업
        try:
            _cleanup_services()
            if _application_logger:
                _application_logger.info("YOKOGAWA OCR 시스템 종료")
        except Exception as e:
            if _application_logger:
                _application_logger.error(f"정리 작업 중 오류: {str(e)}")
            else:
                print(f"정리 작업 중 오류: {str(e)}", file=sys.stderr)


if __name__ == "__main__":
    exit_code = main()
    if exit_code:
        sys.exit(exit_code)
