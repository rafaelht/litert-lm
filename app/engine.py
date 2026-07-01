from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import platform
import time
from typing import Optional

from litert_lm import Engine
from litert_lm.interfaces import CPU

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_engine_lock = asyncio.Lock()
_engine_just_reloaded: bool = False

# Variables para control de TTL
_last_active_time: float = 0.0
_cleanup_task: Optional[asyncio.Task] = None
TTL_SECONDS: int = 3600  # 1 hora en reposo antes de descargar


def _get_max_num_tokens_override() -> int | None:
    raw = os.getenv("LITERT_MAX_NUM_TOKENS", "").strip()
    if not raw:
        return None

    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid LITERT_MAX_NUM_TOKENS value '%s'. Ignoring override.", raw)
        return None

    if value <= 0:
        logger.warning("LITERT_MAX_NUM_TOKENS must be > 0. Ignoring override: %s", value)
        return None

    return value


def force_garbage_collection() -> None:
    """
    Devuelve memoria al sistema operativo cuando la plataforma lo soporta.

    malloc_trim() solamente existe en glibc (Linux).
    """
    if platform.system() != "Linux":
        return

    try:
        libc = ctypes.CDLL("libc.so.6")
        result = libc.malloc_trim(0)

        if result == 1:
            logger.info("Unused memory returned to the operating system.")

    except Exception as e:
        logger.warning("malloc_trim failed: %s", e)


def update_engine_activity() -> None:
    """Actualiza el timestamp de última actividad."""
    global _last_active_time
    _last_active_time = time.time()


async def _monitor_inactivity() -> None:
    """Descarga automáticamente el modelo cuando expira el TTL."""
    global _engine

    while _engine is not None:
        await asyncio.sleep(15)

        async with _engine_lock:
            if _engine is None:
                break

            elapsed = time.time() - _last_active_time

            if elapsed >= TTL_SECONDS:
                logger.info(
                    "Inactivity TTL reached (%ds). Unloading LiteRT engine...",
                    TTL_SECONDS,
                )

                engine = _engine
                _engine = None

                await asyncio.to_thread(engine.close)

                logger.info("LiteRT engine unloaded due to inactivity.")

                force_garbage_collection()

                break


async def init_engine() -> Engine:
    """Inicializa el motor de manera segura garantizando concurrencia."""
    global _engine, _cleanup_task, _engine_just_reloaded

    if _engine is not None:
        update_engine_activity()
        return _engine

    async with _engine_lock:
        if _engine is not None:
            update_engine_activity()
            return _engine

        settings = get_settings()
        max_num_tokens = _get_max_num_tokens_override()
        vision_backend = CPU()
        vision_backend.use_kernel = True

        if max_num_tokens is None:
            logger.info(
                "Initializing LiteRT engine with model at %s (Context: SDK default, vision enabled)",
                settings.model_path,
            )
            _engine = await asyncio.to_thread(
                Engine,
                model_path=settings.model_path,
                vision_backend=vision_backend,
            )
        else:
            logger.info(
                "Initializing LiteRT engine with model at %s (Context override: %d tokens, vision enabled)",
                settings.model_path,
                max_num_tokens,
            )
            _engine = await asyncio.to_thread(
                Engine,
                model_path=settings.model_path,
                vision_backend=vision_backend,
                max_num_tokens=max_num_tokens,
            )
        logger.info("LiteRT engine initialized")

        _engine_just_reloaded = True
        update_engine_activity()

        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = asyncio.create_task(_monitor_inactivity())

        return _engine


def get_engine() -> Engine:
    """Retorna la instancia actual del motor."""
    global _engine

    if _engine is not None:
        update_engine_activity()

    return _engine


def check_and_consume_reload_flag() -> bool:
    """Indica si el motor fue recargado y consume el estado."""
    global _engine_just_reloaded

    if _engine_just_reloaded:
        _engine_just_reloaded = False
        return True

    return False


async def close_engine() -> None:
    """Libera los recursos del motor al apagar el servidor."""
    global _engine

    async with _engine_lock:
        if _engine is None:
            return

        engine = _engine
        _engine = None

        logger.info("Closing LiteRT engine...")

        await asyncio.to_thread(engine.close)

        logger.info("LiteRT engine closed.")

        force_garbage_collection()