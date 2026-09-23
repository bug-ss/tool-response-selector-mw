"""Tests for ToolResponseSelectorMiddleware.

The TypeSafe classifier is replaced by a stub in every test: these cover the
message rewriting, not the quality of Jev's judgments.
"""

from __future__ import annotations

import asyncio

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_typesafe.types import ClassificationResponse, NoulAnswer

from tool_response_selector import (
    ELIDED_CONTENT,
    ToolResponseSelectorMiddleware,
)

LONG = "x" * 500


def answer(probability: float) -> ClassificationResponse:
    return ClassificationResponse(
        model="jev-latest",
        answers={"needed": NoulAnswer(type="noul", noul=probability)},
    )


class StubClassifier:
    """Stands in for TypeSafeClassifier with scripted results, one per state.

    Each result is a probability, an exception, or a ready-made response. The
    last one repeats for any remaining states.
    """

    def __init__(self, *results) -> None:
        self.results = results or (0.5,)
        self.states: list[dict] = []

    def batch(self, states, config=None, *, return_exceptions=False):
        self.states = list(states)
        out = []
        for n in range(len(self.states)):
            result = self.results[min(n, len(self.results) - 1)]
            if isinstance(result, Exception):
                if not return_exceptions:
                    raise result
                out.append(result)
            elif isinstance(result, ClassificationResponse):
                out.append(result)
            else:
                out.append(answer(result))
        return out

    async def abatch(self, states, config=None, *, return_exceptions=False):
        return self.batch(states, config, return_exceptions=return_exceptions)


def build(*results, **kwargs):
    middleware = ToolResponseSelectorMiddleware(**kwargs)
    stub = StubClassifier(*results)
    middleware.classifier = stub
    return middleware, stub


class Request:
    """Minimal stand-in for ModelRequest with the fields the middleware uses."""

    def __init__(self, messages):
        self.messages = messages

    def override(self, **overrides):
        return Request(overrides.get("messages", self.messages))


def run(middleware, messages):
    """Invoke the hook and return the messages that reached the model."""
    seen = {}

    def handler(request):
        seen["messages"] = request.messages
        return AIMessage(content="done")

    middleware.wrap_model_call(Request(messages), handler)
    return seen["messages"]


def conversation(*, turns: int = 2) -> list:
    """Build a run with `turns` tool-calling turns, one tool call each."""
    messages = [HumanMessage(content="Find the config and fix the timeout.")]
    for n in range(turns):
        messages.append(
            AIMessage(
                content="",
                tool_calls=[
                    {"name": f"tool_{n}", "args": {"q": n}, "id": f"call_{n}"}
                ],
            )
        )
        messages.append(
            ToolMessage(content=f"{LONG}{n}", tool_call_id=f"call_{n}", name=f"tool_{n}")
        )
    return messages


def test_elides_result_below_threshold():
    middleware, _ = build(0.1)
    messages = conversation(turns=2)

    selected = run(middleware, messages)

    assert selected[2].content == ELIDED_CONTENT
    assert selected[4].content == messages[4].content


def test_keeps_result_at_or_above_threshold():
    middleware, _ = build(0.9)

    selected = run(middleware, conversation(turns=2))

    assert ELIDED_CONTENT not in [m.content for m in selected]


def test_elided_messages_are_preserved_structurally():
    """Every tool call must keep a matching tool result, or the API rejects it."""
    middleware, _ = build(0.0)
    messages = conversation(turns=3)

    selected = run(middleware, messages)

    assert len(selected) == len(messages)
    assert [type(m) for m in selected] == [type(m) for m in messages]
    called = {c["id"] for m in selected if isinstance(m, AIMessage) for c in m.tool_calls}
    answered = {m.tool_call_id for m in selected if isinstance(m, ToolMessage)}
    assert called == answered
    for i, n in ((2, 0), (4, 1)):
        assert selected[i].content == ELIDED_CONTENT
        assert selected[i].tool_call_id == f"call_{n}"
        assert selected[i].name == f"tool_{n}"


def test_most_recent_turn_is_never_classified():
    middleware, stub = build(0.0)
    messages = conversation(turns=3)

    selected = run(middleware, messages)

    # Two older results classified; the latest turn's result is untouched.
    assert len(stub.states) == 2
    assert selected[6].content == messages[6].content


def test_single_turn_run_is_left_alone():
    middleware, stub = build()

    selected = run(middleware, conversation(turns=1))

    assert stub.states == []
    assert ELIDED_CONTENT not in [m.content for m in selected]


def test_short_results_are_skipped():
    middleware, stub = build(0.0, min_chars=200)
    messages = conversation(turns=2)
    messages[2] = ToolMessage(content="ok", tool_call_id="call_0", name="tool_0")

    selected = run(middleware, messages)

    assert stub.states == []
    assert selected[2].content == "ok"


def test_original_messages_are_not_mutated():
    middleware, _ = build(0.0)
    messages = conversation(turns=2)
    original = messages[2].content

    run(middleware, messages)

    assert messages[2].content == original


def test_classifier_failure_sends_full_context():
    middleware, _ = build(RuntimeError("TypeSafe unavailable"))
    messages = conversation(turns=2)

    selected = run(middleware, messages)

    assert [m.content for m in selected] == [m.content for m in messages]


def test_one_failure_keeps_only_that_result():
    middleware, _ = build(RuntimeError("request too large"), 0.0)
    messages = conversation(turns=3)

    selected = run(middleware, messages)

    assert selected[2].content == messages[2].content
    assert selected[4].content == ELIDED_CONTENT


def test_missing_answer_keeps_result():
    middleware, _ = build(ClassificationResponse(model="jev-latest", answers={}))
    messages = conversation(turns=2)

    selected = run(middleware, messages)

    assert selected[2].content == messages[2].content


def test_classifier_state_carries_task_step_and_call():
    middleware, stub = build(0.9)
    messages = conversation(turns=2)
    messages[3] = AIMessage(
        content="Now reading the config.",
        tool_calls=[{"name": "tool_1", "args": {"q": 1}, "id": "call_1"}],
    )

    run(middleware, messages)

    state = stub.states[0]
    assert state["task"] == "Find the config and fix the timeout."
    assert state["current_step"] == {
        "said": "Now reading the config.",
        "called": [{"name": "tool_1", "args": {"q": 1}}],
    }
    assert state["tool_call"] == {"name": "tool_0", "args": {"q": 0}}
    assert state["tool_result"] == messages[2].content


def test_long_results_are_excerpted_for_the_classifier():
    middleware, stub = build(0.9)
    messages = conversation(turns=2)
    big = "y" * 50_000
    messages[2] = ToolMessage(content=big, tool_call_id="call_0", name="tool_0")

    selected = run(middleware, messages)

    sent = stub.states[0]["tool_result"]
    assert sent.startswith("y" * 1_000)
    assert len(sent) < len(big) // 10
    # The model still gets the full result when it is kept.
    assert selected[2].content == big


class RecordingModel(BaseChatModel):
    """Fake model that records the messages each call receives."""

    replies: list = []
    seen: list = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        reply = self.replies[len(self.seen) - 1]
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def bind_tools(self, tools, **kwargs):
        return self

    @property
    def _llm_type(self) -> str:
        return "recording"


def lookup_agent():
    """Build an agent that looks up "a", then "b", then answers."""
    from langchain.agents import create_agent
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return LONG + q

    model = RecordingModel(
        replies=[
            AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"q": "a"}, "id": "c1"}]
            ),
            AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"q": "b"}, "id": "c2"}]
            ),
            AIMessage(content="All done."),
        ],
        seen=[],
    )
    middleware, _ = build(0.0)
    return create_agent(model, tools=[lookup], middleware=[middleware]), model


def assert_stale_result_elided(model, result):
    # Third model call: the first result is stale and elided, the second is recent.
    third_call = model.seen[2]
    assert ELIDED_CONTENT in [m.content for m in third_call]
    assert sum(m.content == LONG + "b" for m in third_call) == 1
    # Full history survives in agent state.
    assert LONG + "a" in [m.content for m in result["messages"]]
    assert result["messages"][-1].content == "All done."


def test_runs_inside_create_agent():
    agent, model = lookup_agent()

    result = agent.invoke({"messages": [HumanMessage(content="Look up a then b.")]})

    assert_stale_result_elided(model, result)


def test_runs_inside_async_agent():
    agent, model = lookup_agent()

    result = asyncio.run(
        agent.ainvoke({"messages": [HumanMessage(content="Look up a then b.")]})
    )

    assert_stale_result_elided(model, result)
