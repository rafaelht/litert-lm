from __future__ import annotations

import asyncio
import logging
from typing import Optional

from litert_lm import Engine

from app.config import get_settings

logger = logging.getLogger(__name__)

_engine: Optional[Engine] = None
_engine_lock = asyncio.Lock()


async def init_engine() -> Engine:
    global _engine

    if _engine is not None:
        return _engine

    async with _engine_lock:
        if _engine is not None:
            return _engine

        settings = get_settings()
        logger.info("Initializing LiteRT engine with model at %s", settings.model_path)
        _engine = await asyncio.to_thread(Engine, settings.model_path)
        logger.info("LiteRT engine initialized")
        return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Engine is not initialized")
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
