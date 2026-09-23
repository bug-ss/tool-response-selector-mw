# tool-response-selector-mw

LangChain middleware that asks [TypeSafe's Jev](https://docs.langchain.com/oss/python/integrations/providers/typesafe)
which earlier tool results still matter, and sends only those to the model.

A long agent run accumulates tool results that have stopped being useful — a
directory listing consumed three steps ago, a search that returned nothing, a
file read that has already been acted on. They sit in the context window for the
rest of the run, costing input tokens on every model call and competing for the
model's attention.

This middleware runs on `wrap_model_call`. Before each model call it scores the
older tool results with a single `Noul` question — *does the agent still need
this?* — and elides the ones below a probability threshold.

## Usage

```python
from langchain.agents import create_agent
from tool_response_selector import ToolResponseSelectorMiddleware

agent = create_agent(
    "openai:gpt-5.6-terra",
    tools=[...],
    middleware=[ToolResponseSelectorMiddleware(threshold=0.5)],
)
```

Needs `TYPESAFE_API_KEY`. See `example.py` for a runnable script; it also needs
`langchain-openai` and `OPENAI_API_KEY`.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `threshold` | `0.5` | Keep a result when its probability of being needed is at least this. Raise to prune harder. |
| `keep_recent_turns` | `1` | Trailing tool-calling turns exempt from classification. |
| `min_chars` | `200` | Skip results shorter than this. |

## How it behaves

**Elided results are replaced, not deleted.** Provider APIs reject a
conversation whose tool call has no matching tool result, so a dropped
`ToolMessage` would make the request invalid. The message stays, keeping its
`tool_call_id`, with its content swapped for a one-line placeholder.

**Agent state is never modified.** `wrap_model_call` rewrites only the messages
for one model call. The full history stays in state, and each judgment sees the
agent's latest step, so a result elided at one step is reconsidered at the next —
if the work turns back toward it, it comes back.

**The latest turn is exempt.** The results the model just asked for are always
needed; eliding them makes the agent call the same tools again in a loop.
`keep_recent_turns` controls how many trailing turns are protected.

**Classifier failures fail open.** A result whose classification fails is kept,
and the others are still judged; if TypeSafe is unreachable, the full context is
sent. Spending tokens is recoverable; losing context mid-run is not.

## Tradeoffs worth knowing

**It breaks prompt caching.** This is the big one. Providers cache on an exact
message prefix. Rewriting a message in the middle of the history invalidates
everything after it, and the selection changes between calls, so you can end up
paying full price on a prefix that was being served from cache at a 90% discount.
On a run with many turns over a stable context, caching alone may beat this
middleware. It pays off when the context is large, long-lived, and mostly stale —
and when attention, not just cost, is the problem.

**Each model call costs N classification requests.** Each one sends the task, the
agent's latest step, and at most the first 4,000 characters of one result, so
Jev's cost per result stays bounded however large the result is. They run in
parallel, but they add a round trip of latency before each model call.
`min_chars` keeps short results out of the batch.

**Relevance is a judgment call.** Jev returns a calibrated probability, not a
fact. At `threshold=0.5` the middleware will occasionally elide something the
model wanted, and the agent will call the tool again to get it back. For
read-only tools that costs a round trip. For tools with side effects it can
repeat the action: Jev is asked to keep results that record what the agent
created, sent, or changed, but that is a judgment too. Tune `threshold` against
your own runs, and lower it when re-calling a tool is expensive or unsafe.

## Tests

```bash
pytest
```

The classifier is stubbed throughout; the tests cover the message rewriting, not
Jev's judgment. The last two run a real `create_agent` loop, sync and async,
against a fake model and assert that a stale result is elided while the history
stays intact.
