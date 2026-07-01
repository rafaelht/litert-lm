from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import get_settings
from app.conversation_manager import get_conversation_manager
from app.conversation_synchronizer import (
    ConversationSynchronizer,
    apply_snapshot_to_state,
    build_completed_history,
)
from app.engine import get_engine, init_engine, update_engine_activity, check_and_consume_reload_flag
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    OpenAIModel,
    OpenAIModelListResponse,
)
from app.tools import build_runtime_tools
from app.utils import (
    extract_api_key,
    extract_incremental_message,
    make_conversation_id,
    normalize_text_content,
    sdk_message_to_text,
    stable_json_hash,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compatible"])

def _sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def _estimate_token_count(text: str) -> int:
    if not text:
        return 0

    try:
        engine = get_engine()
        if engine is None:
            return 0
        tokens = await asyncio.to_thread(engine.tokenize, text)
        return len(tokens) if isinstance(tokens, list) else 0
    except Exception:
        return 0


def _generate_heuristic_title(prompt: str) -> str:
    """Genera un título dinámico, limpio y ultra-rápido basado en el texto del usuario."""
    if not prompt:
        return "Conversación General"
    
    clean_text = prompt.replace('"', '').replace("'", "").replace("`", "").strip()
    lines = [line.strip() for line in clean_text.splitlines() if line.strip()]
    first_line = lines[0] if lines else clean_text

    words = first_line.split()
    if not words:
        return "Conversación General"

    title_words = words[:4]
    title = " ".join(title_words)

    if len(title) > 30:
        title = title[:27] + "..."
    
    return title.strip().capitalize()


@router.get("/models", response_model=OpenAIModelListResponse)
async def list_models() -> OpenAIModelListResponse:
    settings = get_settings()
    return OpenAIModelListResponse(
        data=[
            OpenAIModel(
                id=settings.model_id,
                created=int(time.time()),
            )
        ]
    )


@router.post("/chat/completions", response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
    raw_request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    request_extra = request.model_extra or {}
    request_tools = request_extra.get("tools")
    request_tool_choice = request_extra.get("tool_choice")
    runtime_tools = build_runtime_tools(request_tools)
    automatic_tool_calling = not bool(runtime_tools)

    message_dicts = [
        message.model_dump(by_alias=True, exclude_none=True)
        for message in request.messages
    ]
    if not message_dicts:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    logger.info("[DEBUG] Payload messages count: %d", len(message_dicts))

    incremental_message = extract_incremental_message(message_dicts)
    incremental_text = normalize_text_content(incremental_message.get("content", ""))
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # 1. CORTOCIRCUITO: Intercepción de prompts administrativos (Títulos y Tags)
    msg_lower = incremental_text.lower()
    
    is_title_req = (
        "title" in msg_lower or 
        "creative title" in msg_lower or
        "phrase with an emoji" in msg_lower or
        (request.max_tokens is not None and request.max_tokens <= 24)
    )
    
    is_tags_req = (
        "generate 1-3 broad tags" in msg_lower or
        "tags for this conversation" in msg_lower or
        (len(message_dicts) == 1 and "tags" in msg_lower)
    )

    if is_title_req or is_tags_req:
        logger.info("[BYPASS] Interceptada petición administrativa de OpenWebUI.")
        
        if is_title_req:
            chat_title = "Conversación General"
            try:
                user_prompt = ""
                for msg in reversed(message_dicts):
                    content = normalize_text_content(msg.get("content", ""))
                    if content and not any(k in content.lower() for k in ["task:", "generate", "create a concise", "{{prompt"]):
                        user_prompt = content
                        break
                
                if not user_prompt and message_dicts:
                    user_prompt = normalize_text_content(message_dicts[0].get("content", ""))

                if "user:" in user_prompt.lower():
                    user_prompt = user_prompt.lower().split("user:")[-1].strip()

                chat_title = _generate_heuristic_title(user_prompt)
                
            except Exception as e:
                logger.error("[BYPASS ERROR] Error procesando título: %s", str(e))
                chat_title = "Conversación General"
            
            mock_payload = {"title": chat_title}

        else:
            mock_payload = ["Technology", "Code"]

        mock_json = json.dumps(mock_payload, ensure_ascii=False)
        
        if request.stream:
            async def static_stream() -> AsyncIterator[str]:
                yield _sse_data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]
                })
                yield _sse_data({
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [{"index": 0, "delta": {"content": mock_json}, "finish_reason": "stop"}]
                })
                yield _sse_data("[DONE]")
            return StreamingResponse(
                static_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
            )
        else:
            return JSONResponse(content={
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": mock_json},
                    "finish_reason": "stop"
                }]
            })

    # 2. FLUJO NORMAL DE CONVERSACIÓN
    engine_instance = await init_engine()

    api_key = extract_api_key(authorization)
    conversation_id = make_conversation_id(api_key, request.model, message_dicts)
    if runtime_tools:
        conversation_id = stable_json_hash(
            {
                "base": conversation_id,
                "tools": request_tools,
                "tool_choice": request_tool_choice,
            }
        )

    manager = get_conversation_manager()

    if check_and_consume_reload_flag():
        logger.info("Detectada recarga del Engine. Limpiando y reasignando referencias de C++.")
        if hasattr(manager, "_engine"):
            manager._engine = engine_instance
        if hasattr(manager, "_conversations"):
            manager._conversations.clear()
        elif hasattr(manager, "clear"):
            manager.clear()

    sync_result = await ConversationSynchronizer(manager).sync(
        conversation_id,
        message_dicts,
        tools=runtime_tools if runtime_tools else None,
        automatic_tool_calling=automatic_tool_calling,
    )
    state = sync_result.state
    sync_snapshot = sync_result.snapshot
    incremental_message = extract_incremental_message(message_dicts)
    
    update_engine_activity()

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            async with state.lock:
                state.touch()
                update_engine_activity()

                first_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant"},
                            "finish_reason": None,
                        }
                    ],
                }
                yield _sse_data(first_chunk)

                response_parts: list[str] = []
                pending_text_parts: list[str] = []
                last_flush = time.monotonic()

                try:
                    iterator = state.conversation.send_message_async(incremental_message)

                    while True:
                        if await raw_request.is_disconnected():
                            logger.info("Cliente desconectado de OpenWebUI. Abortando stream.")
                            break

                        try:
                            sdk_chunk = await asyncio.to_thread(next, iterator, None)
                        except StopIteration:
                            break

                        if sdk_chunk is None:
                            break

                        state.touch()
                        update_engine_activity()

                        text_piece = sdk_message_to_text(sdk_chunk)
                        tool_calls = sdk_chunk.get("tool_calls") if isinstance(sdk_chunk, dict) else None

                        if tool_calls:
                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"tool_calls": tool_calls},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield _sse_data(payload)
                            continue

                        if not text_piece:
                            continue

                        response_parts.append(text_piece)
                        pending_text_parts.append(text_piece)

                        now = time.monotonic()
                        should_flush = len(pending_text_parts) >= 4 or (now - last_flush) >= 0.02
                        if should_flush:
                            chunk_text = "".join(pending_text_parts)
                            pending_text_parts.clear()
                            last_flush = now
                            if chunk_text:
                                payload = {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": request.model,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"content": chunk_text},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield _sse_data(payload)

                    if pending_text_parts:
                        chunk_text = "".join(pending_text_parts)
                        pending_text_parts.clear()
                        if chunk_text:
                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": chunk_text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield _sse_data(payload)

                except Exception as exc:
                    logger.exception("Streaming failed for conversation %s", conversation_id)
                    err_payload = {
                        "error": {
                            "message": str(exc),
                            "type": "internal_error",
                            "code": None,
                        }
                    }
                    yield _sse_data(err_payload)
                    return
                response_text = "".join(response_parts)
                completed_history = build_completed_history(sync_snapshot, response_text)
                apply_snapshot_to_state(
                    state,
                    sync_snapshot,
                    history_messages=completed_history,
                )

                final_chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                }
                yield _sse_data(final_chunk)
                yield _sse_data("[DONE]")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # Bloque síncrono estándar (No-Stream)
    async with state.lock:
        state.touch()
        update_engine_activity()
        try:
            sdk_response = await asyncio.to_thread(
                state.conversation.send_message,
                incremental_message,
            )
        except Exception as exc:
            logger.exception("Completion failed for conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_text = sdk_message_to_text(sdk_response)
    tool_calls = sdk_response.get("tool_calls") if isinstance(sdk_response, dict) else None
    completed_history = build_completed_history(sync_snapshot, response_text)
    apply_snapshot_to_state(
        state,
        sync_snapshot,
        history_messages=completed_history,
    )

    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)
    prompt_tokens = await _estimate_token_count(prompt_text)
    completion_tokens = await _estimate_token_count(response_text)

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(
                    content=response_text,
                    tool_calls=tool_calls,
                ),
                finish_reason="tool_calls" if tool_calls else "stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))