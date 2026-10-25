import pytest

from analyst.agent import AgentError, ConversationMemory, LLMAnalyst
from analyst.config import Settings
from analyst.models import Turn

from conftest import FakeClient, text_msg, tool_msg


def settings(**kw):
    return Settings(model="test-model", api_key="k", **kw)


def make(store, script, **kw):
    client = FakeClient(script)
    return LLMAnalyst(store, settings(**kw), client=client), client


def test_tool_loop_runs_sql_then_answers(store):
    sql = "SELECT region, SUM(revenue) AS r FROM orders GROUP BY 1 ORDER BY 2 DESC LIMIT 1"
    agent, client = make(store, [tool_msg("run_sql", {"sql": sql, "purpose": "top region"}),
                                 text_msg("**Answer:** East leads.")])
    events = []
    resp = agent.run("Which region has the highest revenue?", ConversationMemory(), events.append)
    assert resp.mode == "llm" and "East" in resp.text
    assert resp.usage == {"input_tokens": 30, "output_tokens": 13, "tool_calls": 1}
    assert resp.artifacts.code[0]["code"] == sql
    # second request must contain the tool_result with real data
    second = client.calls[1]["messages"]
    assert second[-1]["content"][0]["type"] == "tool_result" and "East" in second[-1]["content"][0]["content"]
    assert any(e["type"] == "reset" for e in events) and any(e["type"] == "text" for e in events)


def test_schema_is_in_cacheable_system_prompt(store):
    agent, client = make(store, [text_msg("hi")])
    agent.run("hello", ConversationMemory())
    system = client.calls[0]["system"]
    assert system[0]["cache_control"] == {"type": "ephemeral"} and "TABLE orders" in system[0]["text"]
    assert client.calls[0]["model"] == "test-model"


def test_tool_error_is_fed_back_so_model_can_self_correct(store):
    agent, client = make(store, [tool_msg("run_sql", {"sql": "SELECT bad FROM orders", "purpose": "x"}),
                                 tool_msg("run_sql", {"sql": "SELECT COUNT(*) AS n FROM orders", "purpose": "y"}, tool_id="tu_2"),
                                 text_msg("There are 4,212 orders.")])
    resp = agent.run("how many orders?", ConversationMemory())
    first_result = client.calls[1]["messages"][-1]["content"][0]
    assert first_result["is_error"] and first_result["content"].startswith("ERROR")
    assert resp.text == "There are 4,212 orders." and resp.usage["tool_calls"] == 2


def test_step_limit_stops_runaway_loops(store):
    script = [tool_msg("run_sql", {"sql": "SELECT 1 AS a", "purpose": "loop"}, tool_id=f"t{i}") for i in range(5)]
    agent, _ = make(store, script, max_agent_steps=3)
    resp = agent.run("loop forever", ConversationMemory())
    assert "step limit" in resp.text and resp.usage["tool_calls"] == 3


def test_api_failure_becomes_agent_error(store):
    agent, _ = make(store, [RuntimeError("boom")])
    with pytest.raises(AgentError, match="boom"):
        agent.run("q", ConversationMemory())


def test_missing_api_key_raises(store):
    with pytest.raises(AgentError):
        LLMAnalyst(store, Settings(model="m", api_key=None))


def test_conversation_context_is_replayed(store):
    mem = ConversationMemory()
    mem.add(Turn("top region?", "East.", "-- top region\nSELECT region ..."))
    agent, client = make(store, [text_msg("And by product: Beta Phone.")])
    agent.run("and by product?", mem)
    msgs = client.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[0]["content"] == "top region?" and "SELECT region" in msgs[1]["content"]
    assert msgs[2]["content"] == "and by product?"


def test_memory_window():
    mem = ConversationMemory(max_turns=2)
    for i in range(5):
        mem.add(Turn(f"q{i}", f"a{i}"))
    msgs = mem.as_messages()
    assert len(msgs) == 4 and msgs[0]["content"] == "q3"


def test_empty_model_reply_gets_fallback_text(store):
    agent, _ = make(store, [text_msg("")])
    assert "rephrase" in agent.run("?", ConversationMemory()).text
