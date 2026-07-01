from __future__ import annotations

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


def build_runtime_tools(openai_tools: Any) -> list[Tool]:
    if not isinstance(openai_tools, list):
        return []

    runtime_tools: list[Tool] = []
    for item in openai_tools:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function":
            continue
        function_payload = item.get("function")
        if not isinstance(function_payload, dict):
            continue
        if not function_payload.get("name"):
            continue
        runtime_tools.append(OpenAIToolProxy(item))

    return runtime_tools


def get_available_tools() -> list[Tool]:
    # Tools are expected to be provided by the OpenWebUI side.
    return []


def get_tool_functions() -> list[Any]:
    return []
