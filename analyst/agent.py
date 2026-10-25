"""LLM agent: tool-calling loop (Anthropic Messages API) with streaming, memory and graceful errors."""
from __future__ import annotations

import time
from typing import Callable

from .config import Settings, get_settings
from .data_store import DataStore
from .logging_utils import log_event, timed
from .models import AgentResponse, Turn, TurnArtifacts
from .tools import TOOL_SCHEMAS, ToolExecutor

SYSTEM_PROMPT = """You are an expert business data analyst embedded in an application. The user has uploaded CSV \
files that are loaded as SQL tables (DuckDB dialect). Answer questions about them accurately.

RULES
1. Ground every number in a tool result. Never invent or estimate figures; query first, then explain.
2. Choose the right tool: run_sql for aggregations/joins, run_pandas for things SQL does awkwardly, create_chart \
whenever the user asks to show/plot/visualise or when a chart clearly helps, detect_anomalies for outliers, \
data_quality_report for data issues, forecast for projections.
3. Use the table schema below. Use JOINs along the listed likely joins when you need names or attributes from \
another table. Revenue-type questions should usually exclude nothing unless the user says so, but mention if \
cancelled orders, negative values or duplicates affect the result and could be excluded.
4. If a question is ambiguous (e.g. several candidate measures), pick the most sensible default, say which you \
chose and why, and offer the alternative. Only ask a clarifying question if you truly cannot proceed.
5. Follow-ups ("and by product?", "now only 2025", "generate the SQL for that") refer to earlier turns; reuse \
earlier context. If the user asks for SQL or code, show the exact query you ran in a fenced code block.
6. If a tool returns ERROR, fix the query and retry (up to a few times). If the data cannot answer the question, \
say so plainly.
7. Keep answers business-friendly and concise.

ANSWER FORMAT
**Answer:** one or two sentences with the key result (include the actual numbers).
**Insights:** 2-4 short bullets (patterns, shares, comparisons, caveats).
**How I got this:** 1-3 sentences on method, filters, assumptions and any data caveats.
Do not paste large tables into the text; the UI already displays tool outputs and charts.

LOADED DATA
{schema}
"""


class AgentError(RuntimeError):
    """The LLM backend failed (network, auth, quota). The UI falls back to the offline engine."""


class ConversationMemory:
    """Keeps completed turns; replays them compactly (text + the code that was run) on each request."""

    def __init__(self, max_turns: int = 10) -> None:
        self.turns: list[Turn] = []
        self.max_turns = max_turns

    def add(self, turn: Turn) -> None:
        self.turns.append(turn)

    def clear(self) -> None:
        self.turns.clear()

    def as_messages(self) -> list[dict]:
        msgs: list[dict] = []
        for t in self.turns[-self.max_turns:]:
            msgs.append({"role": "user", "content": t.question})
            answer = t.answer + (f"\n\n[Code run for this answer]\n{t.code_summary}" if t.code_summary else "")
            msgs.append({"role": "assistant", "content": answer})
        return msgs


def summarise_code(art: TurnArtifacts, limit: int = 1500) -> str:
    text = "\n".join(f"-- {c['purpose']}\n{c['code']}" for c in art.code)
    return text[:limit]


class LLMAnalyst:
    def __init__(self, store: DataStore, settings: Settings | None = None, client=None) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.client = client or self._make_client()

    def _make_client(self):
        if not self.settings.api_key:
            raise AgentError("No ANTHROPIC_API_KEY configured.")
        import anthropic  # imported lazily so the offline mode has no hard dependency on a key
        return anthropic.Anthropic(api_key=self.settings.api_key, max_retries=2, timeout=90.0)

    def _system(self) -> list[dict]:
        # The schema block is large and stable across turns -> mark it cacheable (prompt caching).
        return [{"type": "text", "text": SYSTEM_PROMPT.format(schema=self.store.schema_text()),
                 "cache_control": {"type": "ephemeral"}}]

    def run(self, question: str, memory: ConversationMemory,
            on_event: Callable[[dict], None] | None = None) -> AgentResponse:
        emit = on_event or (lambda e: None)
        art = TurnArtifacts()
        executor = ToolExecutor(self.store, art)
        messages = memory.as_messages() + [{"role": "user", "content": question}]
        usage = {"input_tokens": 0, "output_tokens": 0, "tool_calls": 0}
        start = time.perf_counter()
        final_text = ""

        with timed("agent.run", mode="llm", question=question[:200]):
            for step in range(self.settings.max_agent_steps):
                emit({"type": "status", "text": "Thinking..." if step == 0 else "Analysing results..."})
                message = self._call(messages, emit)
                usage["input_tokens"] += getattr(message.usage, "input_tokens", 0) or 0
                usage["output_tokens"] += getattr(message.usage, "output_tokens", 0) or 0

                blocks = []
                for b in message.content:
                    if b.type == "text":
                        blocks.append({"type": "text", "text": b.text})
                    elif b.type == "tool_use":
                        blocks.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
                tool_uses = [b for b in blocks if b["type"] == "tool_use"]
                final_text = "".join(b["text"] for b in blocks if b["type"] == "text")

                if not tool_uses:
                    break
                emit({"type": "reset"})   # discard interim "let me check..." text; the final answer follows
                messages.append({"role": "assistant", "content": blocks})
                results = []
                for tu in tool_uses:
                    usage["tool_calls"] += 1
                    emit({"type": "status", "text": f"Running {tu['name']}..."})
                    out = executor.execute(tu["name"], tu["input"])
                    results.append({"type": "tool_result", "tool_use_id": tu["id"], "content": out[:12000],
                                    "is_error": out.startswith("ERROR")})
                messages.append({"role": "user", "content": results})
            else:
                final_text = (final_text + "\n\n_(I stopped after reaching the step limit; "
                              "ask a narrower question for a fuller answer.)_").strip()

        if not final_text.strip():
            final_text = "I could not produce an answer for that. Could you rephrase or be more specific?"
        elapsed = time.perf_counter() - start
        log_event("agent.done", mode="llm", latency_s=round(elapsed, 2), **usage)
        return AgentResponse(text=final_text, artifacts=art, mode="llm", latency_s=elapsed, usage=usage)

    def _call(self, messages: list[dict], emit: Callable[[dict], None]):
        try:
            with self.client.messages.stream(model=self.settings.model, max_tokens=self.settings.max_output_tokens,
                                             system=self._system(), tools=TOOL_SCHEMAS, messages=messages) as stream:
                for delta in stream.text_stream:
                    emit({"type": "text", "delta": delta})
                return stream.get_final_message()
        except AgentError:
            raise
        except Exception as exc:  # SDK raises several typed errors; surface one clean error type
            raise AgentError(f"{type(exc).__name__}: {exc}") from exc
