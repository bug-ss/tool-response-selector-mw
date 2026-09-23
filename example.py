"""Run an agent that prunes stale tool results from each model call.

Requires TYPESAFE_API_KEY and OPENAI_API_KEY.

    pip install langchain-openai
    python example.py
"""

import logging

from langchain.agents import create_agent
from langchain_core.tools import tool

from tool_response_selector import ToolResponseSelectorMiddleware

logging.basicConfig(format="%(message)s")
logging.getLogger("tool_response_selector").setLevel(logging.DEBUG)


@tool
def list_logs(service: str) -> str:
    """List available log files for a service."""
    return "\n".join(f"{service}-2026-09-{day:02d}.log" for day in range(1, 31))


@tool
def read_log(name: str) -> str:
    """Read a log file."""
    return f"{name}\n" + "\n".join(
        f"12:{m:02d}:00 WARN pool exhausted, waited 3000ms" for m in range(60)
    )


@tool
def get_config(service: str) -> str:
    """Get the current configuration for a service."""
    return f'{{"service": "{service}", "pool_size": 8, "timeout_ms": 3000}}'


agent = create_agent(
    "openai:gpt-5.6-terra",
    tools=[list_logs, read_log, get_config],
    middleware=[ToolResponseSelectorMiddleware(threshold=0.5)],
)

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "Checkout is timing out. Find the log file for September 14, "
                    "read it, then tell me what to change in the config."
                ),
            }
        ]
    }
)

print("\n---")
print(result["messages"][-1].content)
