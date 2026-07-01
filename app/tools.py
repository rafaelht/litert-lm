from __future__ import annotations

from typing import Any

from litert_lm.interfaces import Tool


def get_available_tools() -> list[Tool]:
    # Tools are expected to be provided by the OpenWebUI side.
    return []


def get_tool_functions() -> list[Any]:
    return []
