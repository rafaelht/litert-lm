from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from anyio import to_thread

from app.config import get_settings
from app.conversation_manager import get_conversation_manager
from app.engine import get_engine
from app.schemas import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    OpenAIModel,
    OpenAIModelListResponse,
)
from app.utils import (
    bootstrap_messages,
    extract_api_key,
    extract_incremental_message,
    make_conversation_id,
    normalize_text_content,
    sdk_message_to_text,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/v1", tags=["openai-compatible"])


def _sse_data(payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _estimate_token_count(text: str) -> int:
    engine = get_engine()
    if not text:
        return 0

    try:
        tokens = engine.tokenize(text)
        return len(tokens) if isinstance(tokens, list) else 0
    except Exception:
        return 0


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
    message_dicts = [
        message.model_dump(by_alias=True, exclude_none=True)
        for message in request.messages
    ]
    if not message_dicts:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    logger.info("[DEBUG] Payload messages count: %d", len(message_dicts))

    incremental_message = extract_incremental_message(message_dicts).strip()
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    # 1. CORTOCIRCUITO: Detección robusta de prompts administrativos (Títulos y Tags)
    msg_lower = incremental_message.lower()
    
    is_title_req = (
        "title" in msg_lower or 
        "creative title" in msg_lower or
        (request.max_tokens and request.max_tokens <= 16)
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
                # Reconstruir el historial limpio de la conversación enviada por OpenWebUI
                cleaned_history = []
                for msg in message_dicts:
                    role = msg.get("role", "user")
                    content = normalize_text_content(msg.get("content", ""))
                    # Ignorar el prompt del sistema de OpenWebUI dentro del historial
                    if "task:" in content.lower() or "generate a" in content.lower():
                        continue
                    cleaned_history.append(f"{role.upper()}: {content}")
                
                history_str = "\n".join(cleaned_history[-4:])  # Tomar los últimos giros para contexto rápido

                api_key = extract_api_key(authorization)
                manager = get_conversation_manager()
                
                # Forzar un ID único y efímero en el manager para no tocar el KV-Cache del chat real
                title_conv_id = f"system_title_gen_{uuid.uuid4().hex}"
                title_state = await manager.get_or_create(title_conv_id, bootstrap_messages=[])
                
                async with title_state.lock:
                    title_state.touch()
                    
                    # Prompt hiper-restrictivo directo para Gemma
                    title_prompt = (
                        "Eres un asignador de títulos profesional.\n"
                        "Reglas estrictas:\n"
                        "- Devuelve ÚNICAMENTE el título.\n"
                        "- Máximo 4 palabras.\n"
                        "- Sin comillas, sin puntos, sin JSON, sin Markdown.\n"
                        f"Conversación:\n{history_str}\n"
                        "Título:"
                    )
                    
                    sdk_title_response = await asyncio.to_thread(
                        title_state.conversation.send_message,
                        title_prompt,
                    )
                    
                    extracted_title = sdk_message_to_text(sdk_title_response).strip()
                    if extracted_title:
                        # Limpieza extrema por si el modelo ignora instrucciones
                        chat_title = extracted_title.replace('"', '').replace("'", "").replace('\n', '').strip()
                        if "{" in chat_title:
                            chat_title = chat_title.split(":")[-1].replace("}", "").replace('"', '').strip()
                
                # Destrucción inmediata del estado temporal del título
                if title_conv_id in manager._states:
                    del manager._states[title_conv_id]
                    
            except Exception as e:
                logger.error("[BYPASS ERROR] Error en generación de título: %s", str(e))
            
            # Formatear la respuesta exactamente como JSON Object nativo (Lo que OpenWebUI espera)
            mock_payload = {"title": chat_title}

        else:
            # Cortocircuito inmediato para Tags (Evita inferencia innecesaria)
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

    # 2. FLUJO NORMAL DE CONVERSACIÓN (Mantiene el KV-Cache intacto)
    api_key = extract_api_key(authorization)
    conversation_id = make_conversation_id(api_key, request.model, message_dicts)
    manager = get_conversation_manager()

    state = await manager.get_or_create(
        conversation_id,
        bootstrap_messages=bootstrap_messages(message_dicts),
    )

    if request.stream:

        async def event_stream() -> AsyncIterator[str]:
            async with state.lock:
                state.touch()

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

                try:
                    iterator = state.conversation.send_message_async(incremental_message)
                    
                    while True:
                        disconnected = await raw_request.is_disconnected()

                        try:
                            sdk_chunk = await to_thread.run_sync(next, iterator, None)
                            if sdk_chunk is None:
                                break
                        except StopIteration:
                            break

                        state.touch()
                        
                        if not disconnected:
                            text_piece = sdk_message_to_text(sdk_chunk)
                            if not text_piece:
                                continue

                            payload = {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text_piece},
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
        try:
            sdk_response = await asyncio.to_thread(
                state.conversation.send_message,
                incremental_message,
            )
        except Exception as exc:
            logger.exception("Completion failed for conversation %s", conversation_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    response_text = sdk_message_to_text(sdk_response)

    prompt_text = "\n".join(normalize_text_content(msg.get("content")) for msg in message_dicts)
    prompt_tokens = _estimate_token_count(prompt_text)
    completion_tokens = _estimate_token_count(response_text)

    response = ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=request.model,
        choices=[
            ChatCompletionChoice(
                message=ChatCompletionMessage(content=response_text),
                finish_reason="stop",
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return JSONResponse(content=response.model_dump(by_alias=True, exclude_none=True))