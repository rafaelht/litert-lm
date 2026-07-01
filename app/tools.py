from __future__ import annotations

import os
from typing import Any

from litert_lm.interfaces import Tool


class OpenAIToolProxy(Tool):
    def __init__(self, schema: dict[str, Any]) -> None:
        self._schema = schema

    def get_tool_description(self) -> dict[str, Any]:
        return self._schema

    def execute(self, param: dict[str, Any]) -> Any:
        _ = param
        return {
            "error": "external_tool_execution_expected",
            "message": "Tool execution should be performed by the OpenWebUI client.",
        }


def _max_tool_count() -> int:
    raw = os.getenv("MAX_RUNTIME_TOOLS", "8").strip()
    try:
        value = int(raw)
    except ValueError:
        return 8
    return max(1, min(value, 32))


def _shorten_text(value: Any, max_len: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _compact_parameters(parameters: Any) -> dict[str, Any]:
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        return {"type": "object", "properties": {}}

    compact_properties: dict[str, Any] = {}
    for prop_name, prop_schema in list(properties.items())[:20]:
        if not isinstance(prop_schema, dict):
            continue

        compact_prop: dict[str, Any] = {}
        prop_type = prop_schema.get("type")
        if isinstance(prop_type, str):
            compact_prop["type"] = prop_type

        prop_desc = prop_schema.get("description")
        if prop_desc:
            compact_prop["description"] = _shorten_text(prop_desc, 160)

        enum_values = prop_schema.get("enum")
        if isinstance(enum_values, list) and enum_values:
            compact_prop["enum"] = enum_values[:12]

        compact_properties[str(prop_name)] = compact_prop

    required = parameters.get("required")
    compact_required: list[str] = []
    if isinstance(required, list):
        valid_names = set(compact_properties.keys())
        compact_required = [str(name) for name in required if str(name) in valid_names][:20]

    compact: dict[str, Any] = {
        "type": "object",
        "properties": compact_properties,
    }
    if compact_required:
        compact["required"] = compact_required

    return compact


def _compact_tool_schema(item: dict[str, Any]) -> dict[str, Any] | None:
    function_payload = item.get("function")
    if not isinstance(function_payload, dict):
        return None

    name = function_payload.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    compact_function = {
        "name": name.strip(),
        "description": _shorten_text(function_payload.get("description"), 260),
        "parameters": _compact_parameters(function_payload.get("parameters")),
    }
    return {
        "type": "function",
        "function": compact_function,
    }


def build_runtime_tools(openai_tools: Any) -> list[Tool]:
    if not isinstance(openai_tools, list):
        return []

    runtime_tools: list[Tool] = []
    for item in openai_tools:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function":
            continue

        compact = _compact_tool_schema(item)
        if compact is None:
            continue

        runtime_tools.append(OpenAIToolProxy(compact))
        if len(runtime_tools) >= _max_tool_count():
            break

    return runtime_tools


def get_available_tools() -> list[Tool]:
    # Tools are expected to be provided by the OpenWebUI side.
    return []


def get_tool_functions() -> list[Any]:
    return []
