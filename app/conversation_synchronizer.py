from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from app.conversation_manager import ConversationManager, ConversationState
from app.utils import (
    build_context_prompt,
    extract_memory_context,
    extract_system_prompt,
    hash_sdk_messages,
    normalize_text_content,
    sdk_bootstrap_messages,
    stable_json_hash,
    text_hash,
)

logger = logging.getLogger(__name__)

SyncReason = Literal[
    "created",
    "cache_hit",
    "system_changed",
    "memory_changed",
    "history_changed",
]


@dataclass(frozen=True)
class ConversationSnapshot:
    system_prompt: str
    system_hash: str
    memory: str
    memory_hash: str
    sdk_system_message: str
    context_messages: list[dict[str, str]]
    history_hash: str
    message_count: int
    last_message_hash: str
    last_role: str
    last_content: str
    last_message: dict[str, Any]


@dataclass(frozen=True)
class SyncResult:
    state: ConversationState
    snapshot: ConversationSnapshot
    reason: SyncReason
    cache_hit: bool

    @property
    def rebuilt(self) -> bool:
        return self.reason in {"created", "system_changed", "memory_changed", "history_changed"}


class ConversationSynchronizer:
    def __init__(self, manager: ConversationManager) -> None:
        self._manager = manager

    async def sync(
        self,
        conversation_id: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[Any] | None = None,
        automatic_tool_calling: bool = True,
    ) -> SyncResult:
        snapshot = build_snapshot(messages)
        state = await self._manager.get(conversation_id)

        if state is None:
            state = await self._manager.get_or_create(
                conversation_id,
                bootstrap_messages=snapshot.context_messages,
                system_message=snapshot.sdk_system_message or None,
                tools=tools,
                automatic_tool_calling=automatic_tool_calling,
            )
            apply_snapshot_to_state(state, snapshot)
            logger.info("Conversation sync created: %s", conversation_id)
            return SyncResult(state=state, snapshot=snapshot, reason="created", cache_hit=False)

        reason = compare_state(state, snapshot)
        if reason == "cache_hit":
            logger.info("Conversation sync cache hit: %s", conversation_id)
            return SyncResult(state=state, snapshot=snapshot, reason=reason, cache_hit=True)

        state = await self._manager.rebuild(
            conversation_id,
            bootstrap_messages=snapshot.context_messages,
            system_message=snapshot.sdk_system_message or None,
            tools=tools,
            automatic_tool_calling=automatic_tool_calling,
        )
        apply_snapshot_to_state(state, snapshot)
        logger.info("Conversation sync rebuilt %s due to %s", conversation_id, reason)
        return SyncResult(state=state, snapshot=snapshot, reason=reason, cache_hit=False)


def build_snapshot(messages: list[dict[str, Any]]) -> ConversationSnapshot:
    system_prompt = extract_system_prompt(messages)
    memory = extract_memory_context(messages)
    sdk_system_message = build_context_prompt(messages)
    context_messages = sdk_bootstrap_messages(messages)

    last_message = messages[-1] if messages else {}
    last_role = str(last_message.get("role", ""))
    last_content = normalize_text_content(last_message.get("content", "")).strip()
    last_message_payload = {
        "role": last_role,
        "content": last_message.get("content", "") if last_role else "",
    }

    return ConversationSnapshot(
        system_prompt=system_prompt,
        system_hash=text_hash(system_prompt),
        memory=memory,
        memory_hash=text_hash(memory),
        sdk_system_message=sdk_system_message,
        context_messages=context_messages,
        history_hash=hash_sdk_messages(context_messages),
        message_count=len(context_messages),
        last_message_hash=stable_json_hash({"role": last_role, "content": last_content}),
        last_role=last_role,
        last_content=last_content,
        last_message=last_message_payload,
    )


def compare_state(state: ConversationState, snapshot: ConversationSnapshot) -> SyncReason:
    if state.system_hash and state.system_hash != snapshot.system_hash:
        return "system_changed"

    if state.memory_hash and state.memory_hash != snapshot.memory_hash:
        return "memory_changed"

    if state.history_hash == snapshot.history_hash:
        return "cache_hit"

    return "history_changed"


def apply_snapshot_to_state(
    state: ConversationState,
    snapshot: ConversationSnapshot,
    *,
    history_messages: list[dict[str, Any]] | None = None,
) -> None:
    state.system_prompt = snapshot.system_prompt
    state.system_hash = snapshot.system_hash
    state.memory = snapshot.memory
    state.memory_hash = snapshot.memory_hash

    if history_messages is None:
        state.history_hash = snapshot.history_hash
        state.message_count = snapshot.message_count
    else:
        state.history_hash = hash_sdk_messages(history_messages)
        state.message_count = len(history_messages)

    state.last_message_hash = snapshot.last_message_hash
    state.touch()


def build_completed_history(
    snapshot: ConversationSnapshot,
    assistant_response: str,
) -> list[dict[str, Any]]:
    completed = list(snapshot.context_messages)

    if snapshot.last_role == "user" and snapshot.last_content:
        completed.append(snapshot.last_message)

    completed.append({"role": "assistant", "content": assistant_response})
    return completed
