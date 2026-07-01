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


def _get_optimal_threads() -> int:
    """
    Determina automáticamente el número de hilos optimizado para LiteRT-LM (XNNPACK).

    Prioridad:
        1. Variable de entorno LITERT_THREADS.
        2. Detección inteligente de núcleos basada en el diseño de LiteRT (P-Cores vs E-Cores).
        3. Valor por defecto seguro (4).
    """
    env_threads = os.getenv("LITERT_THREADS")

    if env_threads:
        try:
            threads = max(1, int(env_threads))
            logger.info(
                "Using %d threads from LITERT_THREADS environment variable.",
                threads,
            )
            return threads
        except ValueError:
            logger.warning(
                "Invalid LITERT_THREADS value '%s'. Falling back to auto detection.",
                env_threads,
            )

    try:
        import psutil

        physical = psutil.cpu_count(logical=False)

        if physical:
            # Si detecta una CPU con topología masiva (más de 4 núcleos físicos como tu i7),
            # limitar a 4 u 8 previene que los E-Cores ralenticen las barreras síncronas de XNNPACK.
            # 4 es el punto dulce recomendado por Google; 8 es el límite para usar solo P-Cores físicos.
            optimal = 4 if physical <= 4 else 8
            logger.info("Detected %d physical CPU cores. Optimizing backend to %d threads.", physical, optimal)
            return optimal

        logical = psutil.cpu_count(logical=True)
        if logical:
            optimal = 4 if logical <= 4 else 8
            logger.info("Physical core detection unavailable. Using fallback of %d threads.", optimal)
            return optimal

    except Exception as e:
        logger.warning("Unable to detect CPU topology using psutil: %s", e)

    cpu_count = os.cpu_count()
    if cpu_count:
        return 4 if cpu_count <= 4 else 8

    logger.warning("CPU detection failed. Falling back to Google's standard 4 threads.")
    return 4


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
        threads = _get_optimal_threads()

        logger.info(
            "Initializing LiteRT engine with model at %s "
            "(Context: %d tokens, Threads: %d)",
            settings.model_path,
            16384,
            threads,
        )

        cpu_backend = CPU()
        cpu_backend.num_threads = threads

        _engine = await asyncio.to_thread(
            Engine,
            model_path=settings.model_path,
            backend=cpu_backend,
            max_num_tokens=16384,
        )

        logger.info("LiteRT engine initialized successfully.")

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