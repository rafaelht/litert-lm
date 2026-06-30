from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from litert_lm import Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_engine_lock = asyncio.Lock()

# Variables para control de TTL
_last_active_time: float = 0.0
_cleanup_task: Optional[asyncio.Task] = None
TTL_SECONDS: int = 300  # 5 minutos en reposo antes de descargar


def update_engine_activity() -> None:
    """Actualiza el timestamp de última actividad. 
    Llama a esto en tus rutas antes/después de usar el engine.
    """
    global _last_active_time
    _last_active_time = time.time()


async def _monitor_inactivity() -> None:
    """Loop en segundo plano que descarga el modelo si expira el TTL."""
    global _engine
    while _engine is not None:
        await asyncio.sleep(30)  # Verificación cada 30 segundos
        
        async with _engine_lock:
            if _engine is None:
                break
            
            elapsed = time.time() - _last_active_time
            if elapsed >= TTL_SECONDS:
                logger.info("TTL de inactividad alcanzado (%ds). Descargando LiteRT de la RAM...", TTL_SECONDS)
                # Reutilizamos tu función close_engine existente
                engine = _engine
                _engine = None
                await asyncio.to_thread(engine.close)
                logger.info("LiteRT engine liberado automáticamente por inactividad.")
                break


async def init_engine() -> Engine:
    global _engine, _cleanup_task

    if _engine is not None:
        update_engine_activity()
        return _engine

    async with _engine_lock:
        if _engine is not None:
            update_engine_activity()
            return _engine

        settings = get_settings()
        logger.info("Initializing LiteRT engine with model at %s", settings.model_path)
        _engine = await asyncio.to_thread(Engine, settings.model_path)
        logger.info("LiteRT engine initialized")
        
        # Inicializar timers
        update_engine_activity()
        
        # Levantar la tarea de monitoreo en el background del loop de FastAPI
        if _cleanup_task is None or _cleanup_task.done():
            _cleanup_task = asyncio.create_task(_monitor_inactivity())
            
        return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine is not initialized or was unloaded due to inactivity")
    update_engine_activity()
    return _engine


async def close_engine() -> None:
    global _engine

    async with _engine_lock:
        if _engine is None:
            return

        engine = _engine
        _engine = None
        logger.info("Closing LiteRT engine")
        await asyncio.to_thread(engine.close)
        logger.info("LiteRT engine closed")