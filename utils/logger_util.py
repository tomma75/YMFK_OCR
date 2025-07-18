#!/usr/bin/env python3
"""
YOKOGAWA OCR 데이터 준비 프로젝트 - 로깅 유틸리티 모듈

이 모듈은 전체 시스템에서 사용되는 로깅 기능을 제공하며,
구조화된 로깅, 파일 회전, 다중 핸들러 등의 고급 기능을 지원합니다.

작성자: YOKOGAWA OCR 개발팀
작성일: 2025-07-18
"""

import os
import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Callable, TextIO
from pathlib import Path
from enum import Enum
import threading
from queue import Queue
import time

from config.settings import LoggingConfig, ApplicationConfig
from config.constants import (
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
    LOG_FILE_MAX_SIZE_MB,
    LOG_FILE_BACKUP_COUNT,
    LOG_FORMAT_TIMESTAMP,
    LOG_FORMAT_TEMPLATE,
    LOG_ROTATION_WHEN,
    LOG_ROTATION_INTERVAL,
)
from core.exceptions import ApplicationError, ConfigurationError


class LogLevel(Enum):
    """로그 레벨 열거형"""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class LoggerType(Enum):
    """로거 타입 열거형"""

    CONSOLE = "console"
    FILE = "file"
    ROTATING_FILE = "rotating_file"
    DATABASE = "database"
    STRUCTURED = "structured"


# ====================================================================================
# 1. 커스텀 로그 포맷터 클래스들
# ====================================================================================


class LogFormatter(logging.Formatter):
    """
    커스텀 로그 포맷터 클래스

    컬러 코드 및 확장된 정보를 포함한 로그 포맷터입니다.
    """

    # 색상 코드 정의
    COLOR_CODES = {
        "DEBUG": "\033[36m",  # 청록색
        "INFO": "\033[32m",  # 녹색
        "WARNING": "\033[33m",  # 노란색
        "ERROR": "\033[31m",  # 빨간색
        "CRITICAL": "\033[35m",  # 자홍색
        "RESET": "\033[0m",  # 리셋
    }

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        use_colors: bool = True,
        include_traceback: bool = True,
    ):
        """
        LogFormatter 초기화

        Args:
            fmt: 로그 포맷 문자열
            datefmt: 날짜 포맷 문자열
            use_colors: 색상 사용 여부
            include_traceback: 트레이스백 포함 여부
        """
        super().__init__(fmt, datefmt)
        self.use_colors = use_colors
        self.include_traceback = include_traceback

        # 기본 포맷 설정
        if fmt is None:
            fmt = LOG_FORMAT_TEMPLATE
        if datefmt is None:
            datefmt = LOG_FORMAT_TIMESTAMP

        self.default_format = fmt
        self.default_datefmt = datefmt

    def format(self, record: logging.LogRecord) -> str:
        """
        로그 레코드 포맷팅

        Args:
            record: 로그 레코드

        Returns:
            str: 포맷된 로그 메시지
        """
        # 기본 포맷팅
        formatted_message = super().format(record)

        # 색상 적용
        if self.use_colors and hasattr(sys.stdout, "isatty") and sys.stdout.isatty():
            color_code = self.COLOR_CODES.get(record.levelname, "")
            reset_code = self.COLOR_CODES["RESET"]
            formatted_message = f"{color_code}{formatted_message}{reset_code}"

        # 예외 정보 추가
        if record.exc_info and self.include_traceback:
            exc_text = self.formatException(record.exc_info)
            formatted_message += f"\n{exc_text}"

        # 추가 컨텍스트 정보
        if hasattr(record, "context") and record.context:
            context_info = self._format_context(record.context)
            formatted_message += f"\n{context_info}"

        return formatted_message

    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        컨텍스트 정보 포맷팅

        Args:
            context: 컨텍스트 딕셔너리

        Returns:
            str: 포맷된 컨텍스트 정보
        """
        context_lines = []
        for key, value in context.items():
            context_lines.append(f"  {key}: {value}")
        return "Context:\n" + "\n".join(context_lines)


class StructuredLogFormatter(logging.Formatter):
    """
    구조화된 로그 포맷터 클래스

    JSON 형태의 구조화된 로그를 생성합니다.
    """

    def __init__(
        self,
        include_timestamp: bool = True,
        include_level: bool = True,
        include_module: bool = True,
        include_function: bool = True,
        include_line: bool = True,
        extra_fields: Optional[Dict[str, Any]] = None,
    ):
        """
        StructuredLogFormatter 초기화

        Args:
            include_timestamp: 타임스탬프 포함 여부
            include_level: 로그 레벨 포함 여부
            include_module: 모듈명 포함 여부
            include_function: 함수명 포함 여부
            include_line: 라인 번호 포함 여부
            extra_fields: 추가 필드들
        """
        super().__init__()
        self.include_timestamp = include_timestamp
        self.include_level = include_level
        self.include_module = include_module
        self.include_function = include_function
        self.include_line = include_line
        self.extra_fields = extra_fields or {}

    def format(self, record: logging.LogRecord) -> str:
        """
        로그 레코드를 JSON 형태로 포맷팅

        Args:
            record: 로그 레코드

        Returns:
            str: JSON 형태의 로그 메시지
        """
        log_entry = {"message": record.getMessage()}

        # 기본 필드 추가
        if self.include_timestamp:
            log_entry["timestamp"] = datetime.fromtimestamp(record.created).isoformat()

        if self.include_level:
            log_entry["level"] = record.levelname
            log_entry["level_no"] = record.levelno

        if self.include_module:
            log_entry["module"] = record.module
            log_entry["name"] = record.name

        if self.include_function:
            log_entry["function"] = record.funcName

        if self.include_line:
            log_entry["line"] = record.lineno
            log_entry["pathname"] = record.pathname

        # 예외 정보 추가
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info),
            }

        # 추가 컨텍스트 정보
        if hasattr(record, "context") and record.context:
            log_entry["context"] = record.context

        # 사용자 정의 필드 추가
        for key, value in self.extra_fields.items():
            log_entry[key] = value

        # 레코드의 추가 속성들
        for key, value in record.__dict__.items():
            if key not in [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "getMessage",
                "exc_info",
                "exc_text",
                "stack_info",
                "message",
                "context",
            ]:
                log_entry[key] = value

        return json.dumps(log_entry, default=str, ensure_ascii=False)


# ====================================================================================
# 2. 커스텀 핸들러 클래스들
# ====================================================================================


class FileRotatingHandler(logging.handlers.RotatingFileHandler):
    """
    파일 회전 핸들러 클래스

    파일 크기 및 시간 기반 로그 파일 회전을 지원합니다.
    """

    def __init__(
        self,
        filename: str,
        max_size_mb: int = LOG_FILE_MAX_SIZE_MB,
        backup_count: int = LOG_FILE_BACKUP_COUNT,
        encoding: str = "utf-8",
        delay: bool = False,
    ):
        """
        FileRotatingHandler 초기화

        Args:
            filename: 로그 파일명
            max_size_mb: 최대 파일 크기 (MB)
            backup_count: 백업 파일 개수
            encoding: 파일 인코딩
            delay: 파일 생성 지연 여부
        """
        max_bytes = max_size_mb * 1024 * 1024
        super().__init__(
            filename=filename,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding=encoding,
            delay=delay,
        )

        # 로그 파일 디렉터리 생성
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, record: logging.LogRecord) -> None:
        """
        로그 레코드 출력

        Args:
            record: 로그 레코드
        """
        try:
            super().emit(record)
        except Exception as e:
            # 로그 출력 실패 시 콘솔에 에러 메시지 출력
            print(f"로그 출력 실패: {e}", file=sys.stderr)

    def doRollover(self) -> None:
        """파일 회전 수행"""
        try:
            super().doRollover()
            # 회전된 파일의 권한 설정
            if hasattr(self, "stream") and self.stream:
                os.chmod(self.baseFilename, 0o644)
        except Exception as e:
            print(f"파일 회전 실패: {e}", file=sys.stderr)


class DatabaseLogHandler(logging.Handler):
    """
    데이터베이스 로그 핸들러 클래스

    로그를 데이터베이스에 저장합니다.
    """

    def __init__(
        self,
        connection_factory: Callable,
        table_name: str = "application_logs",
        buffer_size: int = 100,
    ):
        """
        DatabaseLogHandler 초기화

        Args:
            connection_factory: 데이터베이스 연결 팩토리
            table_name: 로그 테이블 이름
            buffer_size: 버퍼 크기
        """
        super().__init__()
        self.connection_factory = connection_factory
        self.table_name = table_name
        self.buffer_size = buffer_size
        self.buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()

    def emit(self, record: logging.LogRecord) -> None:
        """
        로그 레코드를 데이터베이스에 저장

        Args:
            record: 로그 레코드
        """
        try:
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created),
                "level": record.levelname,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "message": record.getMessage(),
                "context": getattr(record, "context", None),
            }

            if record.exc_info:
                log_entry["exception"] = traceback.format_exception(*record.exc_info)

            with self.lock:
                self.buffer.append(log_entry)
                if len(self.buffer) >= self.buffer_size:
                    self._flush_buffer()

        except Exception as e:
            print(f"데이터베이스 로그 저장 실패: {e}", file=sys.stderr)

    def _flush_buffer(self) -> None:
        """버퍼의 로그를 데이터베이스에 저장"""
        if not self.buffer:
            return

        try:
            connection = self.connection_factory()
            cursor = connection.cursor()

            for log_entry in self.buffer:
                cursor.execute(
                    f"""
                    INSERT INTO {self.table_name} 
                    (timestamp, level, module, function, line, message, context, exception)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        log_entry["timestamp"],
                        log_entry["level"],
                        log_entry["module"],
                        log_entry["function"],
                        log_entry["line"],
                        log_entry["message"],
                        json.dumps(log_entry.get("context")),
                        json.dumps(log_entry.get("exception")),
                    ),
                )

            connection.commit()
            self.buffer.clear()

        except Exception as e:
            print(f"데이터베이스 로그 플러시 실패: {e}", file=sys.stderr)
        finally:
            if "connection" in locals():
                connection.close()

    def close(self) -> None:
        """핸들러 종료"""
        with self.lock:
            self._flush_buffer()
        super().close()


# ====================================================================================
# 3. 커스텀 로거 클래스
# ====================================================================================


class CustomLogger:
    """
    커스텀 로거 클래스

    애플리케이션 전용 로거 기능을 제공합니다.
    """

    def __init__(
        self,
        name: str,
        config: LoggingConfig,
        logger_type: LoggerType = LoggerType.CONSOLE,
    ):
        """
        CustomLogger 초기화

        Args:
            name: 로거 이름
            config: 로깅 설정
            logger_type: 로거 타입
        """
        self.name = name
        self.config = config
        self.logger_type = logger_type
        self._logger = logging.getLogger(name)
        self._handlers: List[logging.Handler] = []
        self._context: Dict[str, Any] = {}

        # 로거 설정
        self._setup_logger()

    def _setup_logger(self) -> None:
        """로거 설정"""
        # 로그 레벨 설정
        self._logger.setLevel(self.config.get_log_level_numeric())

        # 기존 핸들러 제거
        for handler in self._logger.handlers[:]:
            self._logger.removeHandler(handler)

        # 새 핸들러 설정
        if self.config.enable_console_logging:
            console_handler = self._create_console_handler()
            self._logger.addHandler(console_handler)
            self._handlers.append(console_handler)

        if self.config.enable_file_logging:
            file_handler = self._create_file_handler()
            self._logger.addHandler(file_handler)
            self._handlers.append(file_handler)

        # 상위 로거로의 전파 방지
        self._logger.propagate = False

    def _create_console_handler(self) -> logging.Handler:
        """콘솔 핸들러 생성"""
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, self.config.console_log_level))

        if self.config.enable_structured_logging:
            formatter = StructuredLogFormatter()
        else:
            formatter = LogFormatter(
                fmt=self.config.log_format,
                datefmt=self.config.date_format,
                use_colors=True,
            )

        handler.setFormatter(formatter)
        return handler

    def _create_file_handler(self) -> logging.Handler:
        """파일 핸들러 생성"""
        handler = FileRotatingHandler(
            filename=self.config.log_file_path,
            max_size_mb=self.config.max_file_size_mb,
            backup_count=self.config.backup_count,
        )
        handler.setLevel(getattr(logging, self.config.file_log_level))

        if self.config.enable_structured_logging:
            formatter = StructuredLogFormatter()
        else:
            formatter = LogFormatter(
                fmt=self.config.log_format,
                datefmt=self.config.date_format,
                use_colors=False,
            )

        handler.setFormatter(formatter)
        return handler

    def set_context(self, context: Dict[str, Any]) -> None:
        """
        로깅 컨텍스트 설정

        Args:
            context: 컨텍스트 딕셔너리
        """
        self._context.update(context)

    def clear_context(self) -> None:
        """로깅 컨텍스트 초기화"""
        self._context.clear()

    def _log_with_context(self, level: int, msg: str, *args, **kwargs) -> None:
        """
        컨텍스트와 함께 로그 출력

        Args:
            level: 로그 레벨
            msg: 메시지
            *args: 추가 인수
            **kwargs: 추가 키워드 인수
        """
        if self._context:
            extra = kwargs.get("extra", {})
            extra["context"] = self._context
            kwargs["extra"] = extra

        self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg: str, *args, **kwargs) -> None:
        """디버그 로그"""
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs) -> None:
        """정보 로그"""
        self._log_with_context(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs) -> None:
        """경고 로그"""
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs) -> None:
        """오류 로그"""
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs) -> None:
        """치명적 오류 로그"""
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs) -> None:
        """예외 로그"""
        kwargs["exc_info"] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

    def log_method_entry(self, method_name: str, **kwargs) -> None:
        """메서드 진입 로그"""
        context = {"method": method_name, "event": "method_entry"}
        context.update(kwargs)
        self.debug(f"Entering method: {method_name}", extra={"context": context})

    def log_method_exit(self, method_name: str, **kwargs) -> None:
        """메서드 종료 로그"""
        context = {"method": method_name, "event": "method_exit"}
        context.update(kwargs)
        self.debug(f"Exiting method: {method_name}", extra={"context": context})

    def log_performance(self, operation: str, duration: float, **kwargs) -> None:
        """성능 로그"""
        context = {
            "operation": operation,
            "duration_seconds": duration,
            "event": "performance",
        }
        context.update(kwargs)
        self.info(
            f"Performance: {operation} took {duration:.4f} seconds",
            extra={"context": context},
        )

    def get_logger(self) -> logging.Logger:
        """내부 로거 반환"""
        return self._logger

    def add_handler(self, handler: logging.Handler) -> None:
        """핸들러 추가"""
        self._logger.addHandler(handler)
        self._handlers.append(handler)

    def remove_handler(self, handler: logging.Handler) -> None:
        """핸들러 제거"""
        self._logger.removeHandler(handler)
        if handler in self._handlers:
            self._handlers.remove(handler)

    def close(self) -> None:
        """로거 종료"""
        for handler in self._handlers:
            handler.close()
        self._handlers.clear()


# ====================================================================================
# 4. 로깅 유틸리티 함수들
# ====================================================================================


def setup_logger(name: str, config: LoggingConfig) -> logging.Logger:
    """
    로거 설정 함수
    Args:
        name: 로거 이름
        config: 로깅 설정 객체
    Returns:
        logging.Logger: 설정된 로거
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.log_level))
    
    # 기존 핸들러 제거 (중복 방지)
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # 콘솔 핸들러 추가 (기본)
    if config.enable_console_logging:
        console_handler = create_console_handler()
        console_handler.setFormatter(LogFormatter(config.log_format, config.date_format))
        logger.addHandler(console_handler)
    
    # 파일 핸들러 추가 (옵션)
    if config.enable_file_logging:
        file_handler = create_file_handler(config.log_file_path)
        file_handler.setFormatter(StructuredLogFormatter())  # JSON 형식으로 변경 가능
        logger.addHandler(file_handler)
    
    logger.info(f"Logger '{name}' initialized with config")
    return logger


def create_file_handler(
    log_file_path: str,
    log_level: str = LOG_LEVEL_INFO,
    max_size_mb: int = LOG_FILE_MAX_SIZE_MB,
    backup_count: int = LOG_FILE_BACKUP_COUNT,
    use_structured_format: bool = False,
) -> logging.Handler:
    """
    파일 핸들러 생성

    Args:
        log_file_path: 로그 파일 경로
        log_level: 로그 레벨
        max_size_mb: 최대 파일 크기
        backup_count: 백업 파일 개수
        use_structured_format: 구조화된 포맷 사용 여부

    Returns:
        logging.Handler: 파일 핸들러
    """
    try:
        handler = FileRotatingHandler(
            filename=log_file_path, max_size_mb=max_size_mb, backup_count=backup_count
        )
        handler.setLevel(getattr(logging, log_level.upper()))

        if use_structured_format:
            formatter = StructuredLogFormatter()
        else:
            formatter = LogFormatter(use_colors=False)

        handler.setFormatter(formatter)
        return handler
    except Exception as e:
        raise ConfigurationError(f"Failed to create file handler: {e}")


def create_console_handler(
    log_level: str = LOG_LEVEL_INFO,
    use_colors: bool = True,
    use_structured_format: bool = False,
) -> logging.Handler:
    """
    콘솔 핸들러 생성

    Args:
        log_level: 로그 레벨
        use_colors: 색상 사용 여부
        use_structured_format: 구조화된 포맷 사용 여부

    Returns:
        logging.Handler: 콘솔 핸들러
    """
    try:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper()))

        if use_structured_format:
            formatter = StructuredLogFormatter()
        else:
            formatter = LogFormatter(use_colors=use_colors)

        handler.setFormatter(formatter)
        return handler
    except Exception as e:
        raise ConfigurationError(f"Failed to create console handler: {e}")


def configure_logging(config: LoggingConfig) -> None:
    """
    전체 로깅 시스템 설정

    Args:
        config: 로깅 설정
    """
    try:
        # 루트 로거 설정
        root_logger = logging.getLogger()
        root_logger.setLevel(config.get_log_level_numeric())

        # 기존 핸들러 제거
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 새 핸들러 추가
        if config.enable_console_logging:
            console_handler = create_console_handler(
                log_level=config.console_log_level,
                use_structured_format=config.enable_structured_logging,
            )
            root_logger.addHandler(console_handler)

        if config.enable_file_logging:
            file_handler = create_file_handler(
                log_file_path=config.log_file_path,
                log_level=config.file_log_level,
                max_size_mb=config.max_file_size_mb,
                backup_count=config.backup_count,
                use_structured_format=config.enable_structured_logging,
            )
            root_logger.addHandler(file_handler)

        # 기본 로깅 설정
        logging.basicConfig(
            level=config.get_log_level_numeric(),
            format=config.log_format,
            datefmt=config.date_format,
            handlers=root_logger.handlers,
        )

        # 외부 라이브러리 로깅 레벨 조정
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    except Exception as e:
        raise ConfigurationError(f"Failed to configure logging: {e}")


def get_logger(name: str, config: Optional[LoggingConfig] = None) -> CustomLogger:
    """
    로거 인스턴스 조회

    Args:
        name: 로거 이름
        config: 로깅 설정

    Returns:
        CustomLogger: 로거 인스턴스
    """
    if config is None:
        # 기본 설정 사용
        from config.settings import get_logging_config

        config = get_logging_config()

    return setup_logger(name, config)


def format_log_message(
    message: str,
    level: str,
    timestamp: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
) -> str:
    """
    로그 메시지 포맷팅

    Args:
        message: 메시지
        level: 로그 레벨
        timestamp: 타임스탬프
        context: 컨텍스트 정보

    Returns:
        str: 포맷된 로그 메시지
    """
    if timestamp is None:
        timestamp = datetime.now().strftime(LOG_FORMAT_TIMESTAMP)

    formatted_message = f"[{timestamp}] {level}: {message}"

    if context:
        context_str = json.dumps(context, default=str)
        formatted_message += f" | Context: {context_str}"

    return formatted_message


# ====================================================================================
# 5. 성능 모니터링 데코레이터
# ====================================================================================


def log_execution_time(logger: CustomLogger, operation_name: Optional[str] = None):
    """
    실행 시간 로깅 데코레이터

    Args:
        logger: 로거 인스턴스
        operation_name: 작업 이름
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            logger.log_method_entry(op_name)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_performance(op_name, duration)
                logger.log_method_exit(op_name, success=True)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_performance(op_name, duration, success=False, error=str(e))
                logger.log_method_exit(op_name, success=False, error=str(e))
                raise

        return wrapper

    return decorator


def log_method_calls(logger: CustomLogger):
    """
    메서드 호출 로깅 데코레이터

    Args:
        logger: 로거 인스턴스
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__name__}"
            logger.log_method_entry(
                func_name, args=len(args), kwargs=list(kwargs.keys())
            )

            try:
                result = func(*args, **kwargs)
                logger.log_method_exit(func_name, success=True)
                return result
            except Exception as e:
                logger.log_method_exit(func_name, success=False, error=str(e))
                raise

        return wrapper

    return decorator


# ====================================================================================
# 6. 로깅 매니저 클래스
# ====================================================================================


class LoggingManager:
    """
    로깅 매니저 클래스

    애플리케이션의 모든 로깅을 중앙에서 관리합니다.
    """

    def __init__(self, app_config: ApplicationConfig):
        """
        LoggingManager 초기화

        Args:
            app_config: 애플리케이션 설정
        """
        self.app_config = app_config
        self.logging_config = app_config.logging_config
        self._loggers: Dict[str, CustomLogger] = {}
        self._is_configured = False

    def configure(self) -> None:
        """로깅 시스템 설정"""
        if self._is_configured:
            return

        try:
            configure_logging(self.logging_config)
            self._is_configured = True
        except Exception as e:
            raise ConfigurationError(f"Failed to configure logging manager: {e}")

    def get_logger(self, name: str) -> CustomLogger:
        """
        로거 조회 또는 생성

        Args:
            name: 로거 이름

        Returns:
            CustomLogger: 로거 인스턴스
        """
        if not self._is_configured:
            self.configure()

        if name not in self._loggers:
            self._loggers[name] = setup_logger(name, self.logging_config)

        return self._loggers[name]

    def get_service_logger(self, service_name: str) -> CustomLogger:
        """
        서비스 전용 로거 조회

        Args:
            service_name: 서비스 이름

        Returns:
            CustomLogger: 서비스 로거
        """
        logger_name = f"services.{service_name}"
        return self.get_logger(logger_name)

    def get_util_logger(self, util_name: str) -> CustomLogger:
        """
        유틸리티 전용 로거 조회

        Args:
            util_name: 유틸리티 이름

        Returns:
            CustomLogger: 유틸리티 로거
        """
        logger_name = f"utils.{util_name}"
        return self.get_logger(logger_name)

    def shutdown(self) -> None:
        """로깅 매니저 종료"""
        for logger in self._loggers.values():
            logger.close()
        self._loggers.clear()
        self._is_configured = False


# ====================================================================================
# 7. 모듈 수준 유틸리티
# ====================================================================================

# 전역 로깅 매니저 인스턴스
_logging_manager: Optional[LoggingManager] = None


def initialize_logging(app_config: ApplicationConfig) -> LoggingManager:
    """
    로깅 시스템 초기화

    Args:
        app_config: 애플리케이션 설정

    Returns:
        LoggingManager: 로깅 매니저 인스턴스
    """
    global _logging_manager

    if _logging_manager is None:
        _logging_manager = LoggingManager(app_config)
        _logging_manager.configure()

    return _logging_manager


def get_application_logger(name: str) -> CustomLogger:
    """
    애플리케이션 로거 조회

    Args:
        name: 로거 이름

    Returns:
        CustomLogger: 로거 인스턴스
    """
    if _logging_manager is None:
        # 기본 설정으로 초기화
        from config.settings import load_configuration

        config = load_configuration()
        initialize_logging(config)

    return _logging_manager.get_logger(name)


def shutdown_logging() -> None:
    """로깅 시스템 종료"""
    global _logging_manager

    if _logging_manager is not None:
        _logging_manager.shutdown()
        _logging_manager = None


# ====================================================================================
# 8. 런타임 검증 및 테스트
# ====================================================================================


def validate_logging_configuration(config: LoggingConfig) -> bool:
    """
    로깅 설정 유효성 검증

    Args:
        config: 로깅 설정

    Returns:
        bool: 검증 결과
    """
    try:
        # 로그 레벨 검증
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.log_level not in valid_levels:
            return False

        # 파일 경로 검증
        if config.enable_file_logging:
            log_dir = Path(config.log_file_path).parent
            if not log_dir.exists():
                log_dir.mkdir(parents=True, exist_ok=True)

        # 포맷 검증
        try:
            test_formatter = LogFormatter(
                fmt=config.log_format, datefmt=config.date_format
            )
            test_record = logging.makeLogRecord({"msg": "test"})
            test_formatter.format(test_record)
        except Exception:
            return False

        return True

    except Exception:
        return False


if __name__ == "__main__":
    # 로깅 유틸리티 테스트
    print("YOKOGAWA OCR 로깅 유틸리티 테스트")
    print("=" * 50)

    try:
        # 기본 설정으로 로깅 설정 테스트
        from config.settings import LoggingConfig

        config = LoggingConfig()

        # 설정 검증
        if validate_logging_configuration(config):
            print("✅ 로깅 설정 검증 통과")
        else:
            print("❌ 로깅 설정 검증 실패")

        # 로거 생성 테스트
        test_logger = setup_logger("test_logger", config)
        test_logger.info("테스트 로그 메시지")
        test_logger.debug("디버그 메시지")
        test_logger.warning("경고 메시지")
        test_logger.error("오류 메시지")

        print("✅ 로거 생성 및 로그 출력 테스트 완료")

        # 구조화된 로깅 테스트
        test_logger.set_context({"user_id": "test_user", "session_id": "test_session"})
        test_logger.info("컨텍스트가 포함된 로그 메시지")

        print("✅ 구조화된 로깅 테스트 완료")

        # 성능 로깅 테스트
        @log_execution_time(test_logger, "test_operation")
        def test_function():
            time.sleep(0.1)
            return "완료"

        result = test_function()
        print(f"✅ 성능 로깅 테스트 완료: {result}")

        # 정리
        test_logger.close()

    except Exception as e:
        print(f"❌ 로깅 유틸리티 테스트 실패: {e}")

    print("\n🎯 로깅 유틸리티 구현이 완료되었습니다!")
