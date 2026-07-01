from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from typing import Any

from litert_lm.interfaces import Tool
from litert_lm import tool_from_function


class NasStatusTool(Tool):
    def get_tool_description(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "get_nas_status",
                "description": "Checks whether the NAS is reachable and reports its basic status.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "host": {
                            "type": "string",
                            "description": "Host or IP of the NAS to validate. Optional; defaults to localhost.",
                        }
                    },
                },
            },
        }

    def execute(self, param: dict[str, Any]) -> Any:
        host = str(param.get("host") or "localhost")
        try:
            if platform.system().lower() == "windows":
                result = subprocess.run(
                    ["ping", "-n", "1", host],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            else:
                result = subprocess.run(
                    ["ping", "-c", "1", host],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            return {
                "host": host,
                "reachable": result.returncode == 0,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
            }
        except Exception as exc:  # pragma: no cover - defensive fallback
            return {"host": host, "reachable": False, "error": str(exc)}


def get_available_tools() -> list[Tool]:
    return [NasStatusTool()]


def get_tool_functions() -> list[Any]:
    return [tool_from_function(lambda host="localhost": True)]
