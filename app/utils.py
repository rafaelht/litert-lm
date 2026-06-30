from __future__ import annotations

import hashlib
import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any


def now_ts() -> float:
    return time.time()


class ConversationIdStrategy(ABC):
    @abstractmethod
    def build(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        first_user_message: str,
    ) -> str:
        raise NotImplementedError


class DefaultConversationIdStrategy(ConversationIdStrategy):
    def build(
        self,
        *,
        api_key: str,
        model: str,
        system_prompt: str,
        first_user_message: str,
    ) -> str:
        payload = "\n".join([api_key, model, system_prompt, first_user_message])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


conversation_id_strategy: ConversationIdStrategy = DefaultConversationIdStrategy()


def make_conversation_id(api_key: str, model: str, messages: list[dict[str, Any]]) -> str:
    system_prompt = extract_system_prompt(messages)
    first_user = extract_first_user_message(messages)
    return conversation_id_strategy.build(
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        first_user_message=first_user,
    )


def extract_api_key(auth_header: str | None) -> str:
    if not auth_header:
        return "anonymous"

    parts = auth_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or "anonymous"
    return auth_header.strip() or "anonymous"


def normalize_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in {"text", "input_text"}:
                text_value = part.get("text")
                if isinstance(text_value, str):
                    chunks.append(text_value)
        return "\n".join(chunks)

    if isinstance(content, dict):
        if content.get("type") in {"text", "input_text"} and isinstance(content.get("text"), str):
            return content["text"]

    return ""


def stable_json_hash(payload: Any) -> str:
    normalized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_memory_line(text: str) -> bool:
    lower = text.strip().lower()
    return lower.startswith(
        (
            "memory:",
            "memories:",
            "user memory:",
            "user memories:",
            "memoria:",
            "memorias:",
            "recuerdos:",
        )
    )


def _is_memory_heading(text: str) -> bool:
    lower = text.strip().strip("#*:").lower()
    return lower in {
        "memory",
        "memories",
        "user memory",
        "user memories",
        "memoria",
        "memorias",
        "recuerdos",
    }


def _is_list_item(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(("-", "*", "•")) or re.match(r"^\d+[\).\s]", stripped) is not None


def split_system_and_memory_content(content: str) -> tuple[str, str]:
    system_lines: list[str] = []
    memory_lines: list[str] = []
    in_memory_block = False

    for line in content.splitlines():
        stripped = line.strip()

        if _is_memory_line(stripped) or _is_memory_heading(stripped):
            memory_lines.append(line)
            in_memory_block = True
            continue

        if in_memory_block:
            if not stripped:
                in_memory_block = False
                continue

            if _is_list_item(stripped) or line[:1].isspace():
                memory_lines.append(line)
                continue

            in_memory_block = False

        system_lines.append(line)

    return "\n".join(system_lines).strip(), "\n".join(memory_lines).strip()


def extract_system_prompt(messages: list[dict[str, Any]]) -> str:
    system_parts: list[str] = []
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            content = normalize_text_content(message.get("content"))
            system_content, _ = split_system_and_memory_content(content)
            system_parts.append(system_content)
    return "\n".join([part for part in system_parts if part]).strip()


def extract_raw_context_prompt(messages: list[dict[str, Any]]) -> str:
    context_parts: list[str] = []
    for message in messages:
        if message.get("role") in {"system", "developer"}:
            content = normalize_text_content(message.get("content", "")).strip()
            if content:
                context_parts.append(content)
    return "\n\n".join(context_parts).strip()


def extract_memory_context(messages: list[dict[str, Any]]) -> str:
    memory_parts: list[str] = []
    memory_keys = {"memory", "memories", "user_memory", "user_memories"}

    for message in messages:
        for key in memory_keys:
            value = message.get(key)
            if isinstance(value, str) and value.strip():
                memory_parts.append(value.strip())
            elif isinstance(value, list):
                memory_parts.extend(str(item).strip() for item in value if str(item).strip())

        metadata = message.get("metadata")
        if isinstance(metadata, dict):
            for key in memory_keys:
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    memory_parts.append(value.strip())
                elif isinstance(value, list):
                    memory_parts.extend(str(item).strip() for item in value if str(item).strip())

        if message.get("role") not in {"system", "developer"}:
            continue

        content = normalize_text_content(message.get("content", ""))
        _, memory_content = split_system_and_memory_content(content)
        if memory_content:
            memory_parts.append(memory_content)

    return "\n".join(memory_parts).strip()


def build_context_prompt(messages: list[dict[str, Any]]) -> str:
    return extract_raw_context_prompt(messages)


def extract_first_user_message(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return normalize_text_content(message.get("content")).strip()
    return ""


def extract_incremental_message(messages: list[dict[str, Any]]) -> str:
    if not messages:
        raise ValueError("messages must not be empty")
    return normalize_text_content(messages[-1].get("content", ""))


def bootstrap_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Formatea el historial OpenAI filtrando solo los turnos que LiteRT entiende.
    El system prompt se entrega por separado usando system_message del SDK.
    """
    if len(messages) <= 1:
        return []

    history = messages[:-1]
    bootstrapped: list[dict[str, Any]] = []

    for msg in history:
        role = msg.get("role")
        if role in {"system", "developer"}:
            continue

        content = normalize_text_content(msg.get("content", ""))

        if role in {"user", "assistant"}:
            bootstrapped.append({"role": role, "content": content})

    return bootstrapped


def sdk_bootstrap_messages(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    bootstrapped = bootstrap_messages(messages)
    return [
        {
            "role": str(message.get("role", "")),
            "content": normalize_text_content(message.get("content", "")).strip(),
        }
        for message in bootstrapped
        if message.get("role") in {"user", "assistant"}
    ]


def hash_sdk_messages(messages: list[dict[str, Any]]) -> str:
    normalized = [
        {
            "role": str(message.get("role", "")),
            "content": normalize_text_content(message.get("content", "")).strip(),
        }
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    return stable_json_hash(normalized)


def sdk_message_to_text(message: Any) -> str:
    if isinstance(message, str):
        return message

    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, str):
                    parts.append(item)
            return "".join(parts)
        if isinstance(content, dict) and isinstance(content.get("text"), str):
            return content["text"]

    if hasattr(message, "text"):
        return str(message.text)

    try:
        return json.dumps(message, ensure_ascii=False)
    except Exception:
        return ""
