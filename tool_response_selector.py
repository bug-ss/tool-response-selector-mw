"""Middleware that sends only the relevant tool results to each model call.

Long-running agents accumulate tool results that stop mattering: a directory
listing consumed three steps ago, a search that returned nothing useful, a file
read that has already been acted on. They stay in the context window for the
rest of the run, where they cost input tokens on every subsequent model call and
compete for the model's attention.

`ToolResponseSelectorMiddleware` asks a TypeSafe `Noul` question about each
older tool result -- "does the agent still need this?" -- and elides the ones
that fall below a probability threshold before the request reaches the model.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langchain_typesafe import Noul, NoulCriteria, TypeSafeClassifier
from langchain_typesafe.types import ClassificationResponse

logger = logging.getLogger(__name__)

_QUESTION_ID = "needed"

# Enough of a result to tell what it is. Sending all of it would cost Jev more
# tokens on every call than eliding it saves on the main model.
_EXCERPT_CHARS = 4_000

_MAX_CONCURRENCY = 8

_INSTRUCTIONS = (
    "An AI agent is working on the task in `task`. Its most recent step is "
    "`current_step`: what it said and the tools it called. Earlier in the run it "
    "made the tool call in `tool_call` and got back `tool_result`, which may be "
    "truncated. Does the agent still need `tool_result` in its context for its "
    "next step and the rest of the task?"
)

_CRITERIA = NoulCriteria(
    true=(
        "The result holds specific facts -- identifiers, values, paths, errors, "
        "content -- that the next step or the remaining work depends on, or it "
        "records the outcome of an action the agent took, such as something it "
        "created, sent, or changed."
    ),
    false=(
        "The result is unrelated to the task, `current_step` shows the agent has "
        "already taken what it needed from it and moved on, or it was empty, "
        "failed, or otherwise carried nothing worth keeping."
    ),
)

ELIDED_CONTENT = (
    "[Tool output omitted from this request to save context: it was judged not "
    "needed for the current step. This placeholder is not the tool's output.]"
)


class ToolResponseSelectorMiddleware(AgentMiddleware):
    """Filter stale tool results out of each model call.

    The middleware runs on `wrap_model_call`, so it rewrites only the messages
    sent to the model. Agent state keeps every tool result intact, and because
    each judgment sees the agent's latest step, a result elided from one model
    call is reconsidered on the next one.

    Elided results are replaced by a short placeholder rather than deleted.
    Provider APIs reject a conversation whose tool call has no matching tool
    result, so dropping the message outright would make the request invalid.

    Args:
        threshold: Keep a result when its probability of being needed is at or
            above this value. Raise it to prune harder, lower it to keep more.
        keep_recent_turns: Number of trailing tool-calling turns exempt from
            classification. The results the model just asked for are always
            needed, and eliding them makes the agent call the same tools again.
        min_chars: Skip results shorter than this. Classifying a short result
            can cost more than sending it.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        keep_recent_turns: int = 1,
        min_chars: int = 200,
    ) -> None:
        super().__init__()
        self.threshold = threshold
        self.keep_recent_turns = keep_recent_turns
        self.min_chars = min_chars
        self.classifier = TypeSafeClassifier(
            questions={_QUESTION_ID: Noul(instructions=_INSTRUCTIONS, criteria=_CRITERIA)}
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        candidates = _candidates(request.messages, self.keep_recent_turns, self.min_chars)
        if candidates:
            responses = self.classifier.batch(
                _states(request.messages, candidates),
                config={"max_concurrency": _MAX_CONCURRENCY},
                return_exceptions=True,
            )
            request = self._select(request, candidates, responses)
        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        candidates = _candidates(request.messages, self.keep_recent_turns, self.min_chars)
        if candidates:
            responses = await self.classifier.abatch(
                _states(request.messages, candidates),
                config={"max_concurrency": _MAX_CONCURRENCY},
                return_exceptions=True,
            )
            request = self._select(request, candidates, responses)
        return await handler(request)

    def _select(
        self,
        request: ModelRequest,
        candidates: list[int],
        responses: list[ClassificationResponse | Exception],
    ) -> ModelRequest:
        """Return `request` with the results judged not needed replaced."""
        failures = [r for r in responses if isinstance(r, Exception)]
        if failures:
            # Sending too much context is recoverable; losing it is not.
            logger.warning(
                "TypeSafe classification failed for %d of %d tool results; keeping them",
                len(failures),
                len(responses),
                exc_info=failures[0],
            )

        elide = []
        for i, response in zip(candidates, responses, strict=True):
            if isinstance(response, Exception):
                continue
            answer = response.nouls.get(_QUESTION_ID)
            if answer is not None and answer.noul < self.threshold:
                elide.append(i)
        if not elide:
            return request

        messages = request.messages
        selected = list(messages)
        for i in elide:
            selected[i] = messages[i].model_copy(update={"content": ELIDED_CONTENT})

        logger.debug(
            "Elided %d of %d classified tool results (%d chars)",
            len(elide),
            len(candidates),
            sum(len(messages[i].text) for i in elide),
        )
        return request.override(messages=selected)


def _candidates(
    messages: Sequence[AnyMessage], keep_recent_turns: int, min_chars: int
) -> list[int]:
    """Return indices of the tool results worth classifying."""
    boundary = _recent_turn_boundary(messages, keep_recent_turns)
    return [
        i
        for i, message in enumerate(messages)
        if i < boundary
        and isinstance(message, ToolMessage)
        and len(message.text) >= min_chars
    ]


def _states(messages: Sequence[AnyMessage], candidates: list[int]) -> list[dict[str, Any]]:
    """Build one classifier input per candidate tool result."""
    task = _latest_task(messages)
    current_step = _current_step(messages)
    tool_calls = _tool_calls_by_id(messages)
    return [
        {
            "task": task,
            "current_step": current_step,
            "tool_call": tool_calls.get(messages[i].tool_call_id, {}),
            "tool_result": _excerpt(messages[i].text),
        }
        for i in candidates
    ]


def _recent_turn_boundary(messages: Sequence[AnyMessage], turns: int) -> int:
    """Return the index below which tool results are old enough to classify.

    Walks back over `turns` tool-calling AI messages. Everything from that AI
    message onward belongs to a recent turn and is left alone. If the run has
    not had that many tool-calling turns yet, nothing is classified.
    """
    if turns <= 0:
        return len(messages)
    seen = 0
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, AIMessage) and message.tool_calls:
            seen += 1
            if seen == turns:
                return i
    return 0


def _latest_task(messages: Sequence[AnyMessage]) -> str:
    """Return the most recent human message, which states the current goal."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.text
    return ""


def _current_step(messages: Sequence[AnyMessage]) -> dict[str, Any]:
    """Describe the agent's latest move: what it said and which tools it called."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            return {
                "said": message.text,
                "called": [
                    {"name": call["name"], "args": call["args"]}
                    for call in message.tool_calls
                ],
            }
    return {}


def _excerpt(text: str) -> str:
    """Trim a result to what the classifier needs to recognize it."""
    if len(text) <= _EXCERPT_CHARS:
        return text
    return f"{text[:_EXCERPT_CHARS]}\n[... {len(text) - _EXCERPT_CHARS} more characters]"


def _tool_calls_by_id(messages: Sequence[AnyMessage]) -> dict[str, dict[str, Any]]:
    """Map tool_call_id to the call that produced it, for context when judging."""
    return {
        call["id"]: {"name": call["name"], "args": call["args"]}
        for message in messages
        if isinstance(message, AIMessage)
        for call in message.tool_calls
        if call.get("id")
    }
