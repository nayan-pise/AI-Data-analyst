"""Streamlit UI for the AI Data Analyst.  Run with:  streamlit run app.py"""
from __future__ import annotations

import dataclasses
import hmac
from pathlib import Path

import pandas as pd
import streamlit as st

from analyst.agent import AgentError, ConversationMemory, LLMAnalyst, summarise_code
from analyst.config import get_settings
from analyst.dashboard import build_dashboard
from analyst.data_store import DataStore, DataValidationError
from analyst.logging_utils import METRICS, get_logger, log_event
from analyst.models import AgentResponse, Turn, TurnArtifacts
from analyst.offline import OfflineAnalyst
from analyst.report import build_html, build_markdown

SAMPLE_DIR = Path(__file__).parent / "sample_data"
SUGGESTIONS = [
    "Which region generated the highest revenue?",
    "Show monthly sales trends.",
    "Which products are underperforming?",
    "What are the top five customers?",
    "Detect anomalies in the dataset.",
    "Run a data quality check.",
    "Forecast revenue for the next 6 months.",
    "Generate SQL for this analysis.",
]

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
get_logger(get_settings().log_dir)


# ------------------------------------------------------------------ state
def init_state() -> None:
    ss = st.session_state
    ss.setdefault("store", DataStore())
    ss.setdefault("memory", ConversationMemory(get_settings().max_history_turns))
    ss.setdefault("offline", OfflineAnalyst(ss.store))
    ss.setdefault("messages", [])          # [{"role", "content", "response"}]
    ss.setdefault("loaded_files", set())
    ss.setdefault("authed", False)
    ss.setdefault("notices", [])


def reset_conversation() -> None:
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.session_state.offline = OfflineAnalyst(st.session_state.store)


def password_gate() -> bool:
    pw = get_settings().app_password
    if not pw or st.session_state.authed:
        return True
    st.title("🔒 AI Data Analyst")
    entered = st.text_input("Password", type="password")
    if entered:
        if hmac.compare_digest(entered.encode(), pw.encode()):
            st.session_state.authed = True
            st.rerun()
        st.error("Incorrect password.")
    return False


# ---------------------------------------------------------------- loading
def ingest(uploaded) -> None:
    ss = st.session_state
    for f in uploaded or []:
        key = (f.name, f.size)
        if key in ss.loaded_files:
            continue
        try:
            result = ss.store.load_csv(f.getvalue(), f.name)
            ss.loaded_files.add(key)
            ss.notices.append(("success", f"Loaded **{result.table}**: {result.info.rows:,} rows, {len(result.info.columns)} columns."))
            for w in result.warnings:
                ss.notices.append(("warning", f"{f.name}: {w}"))
        except DataValidationError as exc:
            ss.notices.append(("error", f"{f.name}: {exc}"))
        except Exception as exc:  # never let one bad file crash the app
            log_event("upload_failed", file=f.name, error=str(exc))
            ss.notices.append(("error", f"{f.name}: unexpected error while reading the file ({type(exc).__name__})."))


def load_sample() -> None:
    ss = st.session_state
    if not SAMPLE_DIR.exists():
        ss.notices.append(("error", "Sample data folder not found. Run scripts/generate_sample_data.py."))
        return
    for path in sorted(SAMPLE_DIR.glob("*.csv")):
        key = (path.name, path.stat().st_size)
        if key not in ss.loaded_files:
            result = ss.store.load_csv(path)
            ss.loaded_files.add(key)
            ss.notices.append(("success", f"Loaded sample table **{result.table}** ({result.info.rows:,} rows)."))


def sidebar(settings) -> tuple[str, str | None, str]:
    ss = st.session_state
    with st.sidebar:
        st.title("📊 AI Data Analyst")
        st.caption("Upload CSVs, then ask questions in plain English.")
        uploaded = st.file_uploader("Upload CSV files", type=["csv", "tsv", "txt"], accept_multiple_files=True)
        ingest(uploaded)
        if st.button("Load sample dataset", width="stretch"):
            load_sample()
        for level, msg in ss.notices[-6:]:
            getattr(st, level)(msg)
        ss.notices = ss.notices[-6:]

        if ss.store.tables:
            st.subheader("Loaded tables")
            for t in ss.store.tables.values():
                st.write(f"**{t.name}**  \n{t.rows:,} rows · {len(t.columns)} cols")
            if st.button("Remove all data", width="stretch"):
                for t in list(ss.store.tables):
                    ss.store.drop_table(t)
                ss.loaded_files = set()
                reset_conversation()
                st.rerun()

        st.divider()
        st.subheader("Engine")
        key = st.text_input("Anthropic API key (optional)", type="password", value="",
                            help="Kept only in this session. Without a key the built-in offline engine answers.")
        api_key = key or settings.api_key
        mode = st.radio("Mode", ["Auto", "Offline engine only"], horizontal=False,
                        help="Auto uses the LLM agent when a key is available and falls back to offline on errors.")
        model = st.text_input("Model", value=settings.model)
        st.caption("🟢 LLM agent ready" if api_key and mode == "Auto" else "⚪ Offline engine (no LLM)")
        if st.button("Clear conversation", width="stretch"):
            reset_conversation()
            st.rerun()

        with st.expander("Observability"):
            snap = METRICS.snapshot()
            if snap:
                st.dataframe(pd.DataFrame(snap), hide_index=True, width="stretch")
            st.caption(f"Query cache: {ss.store.cache_hits} hits / {ss.store.cache_misses} misses")
    return mode, api_key, model


# ----------------------------------------------------------------- answering
def answer(question: str, mode: str, api_key: str | None, model: str, settings) -> AgentResponse:
    ss = st.session_state
    if mode == "Auto" and api_key:
        placeholder, status = st.empty(), st.status("Thinking...", expanded=False)
        buf = {"text": ""}

        def on_event(e: dict) -> None:
            if e["type"] == "text":
                buf["text"] += e["delta"]
                placeholder.markdown(buf["text"] + " ▌")
            elif e["type"] == "reset":
                buf["text"] = ""
                placeholder.empty()
            elif e["type"] == "status":
                status.update(label=e["text"])

        try:
            agent = LLMAnalyst(ss.store, dataclasses.replace(settings, api_key=api_key, model=model))
            resp = agent.run(question, ss.memory, on_event)
            status.update(label="Done", state="complete")
            placeholder.empty()
            return resp
        except AgentError as exc:
            status.update(label="LLM unavailable; using offline engine", state="error")
            placeholder.empty()
            log_event("llm_fallback", error=str(exc)[:300])
            resp = ss.offline.run(question, ss.memory)
            resp.warnings.insert(0, f"LLM request failed ({str(exc)[:160]}). Answered with the offline engine instead.")
            return resp
    return ss.offline.run(question, ss.memory)


def render_response(resp: AgentResponse, idx: int) -> None:
    for w in resp.warnings[:1]:
        if "LLM request failed" in w:
            st.warning(w)
    st.markdown(resp.text)
    art = resp.artifacts
    for i, f in enumerate(art.figures):
        st.plotly_chart(f["fig"], width="stretch", key=f"fig-{idx}-{i}")
    if art.tables:
        with st.expander(f"📋 Result tables ({len(art.tables)})", expanded=not art.figures):
            for i, t in enumerate(art.tables):
                st.caption(t["title"])
                st.dataframe(t["df"], hide_index=True, width="stretch", key=f"tbl-{idx}-{i}")
    if art.code:
        with st.expander("🧾 SQL / code used"):
            for c in art.code:
                if c["purpose"]:
                    st.caption(c["purpose"])
                st.code(c["code"], language="sql" if c["language"] == "sql" else "python")
    if art.steps:
        with st.expander("🧠 Reasoning trace"):
            for s in art.steps:
                st.write("• " + s)
    bits = [f"mode: {resp.mode}", f"{resp.latency_s:.1f}s"]
    if resp.usage:
        bits.append(f"{resp.usage.get('input_tokens', 0):,} in / {resp.usage.get('output_tokens', 0):,} out tokens, "
                    f"{resp.usage.get('tool_calls', 0)} tool calls")
    st.caption(" · ".join(bits))


def handle_question(question: str, mode, api_key, model, settings) -> None:
    ss = st.session_state
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            resp = answer(question, mode, api_key, model, settings)
        except Exception as exc:  # last-resort guard so the UI never shows a stack trace
            log_event("answer_failed", error=f"{type(exc).__name__}: {exc}")
            resp = AgentResponse(f"Something went wrong while analysing that ({type(exc).__name__}). Please rephrase and try again.",
                                 TurnArtifacts(), "offline")
        render_response(resp, len(ss.messages))
    ss.messages.append({"role": "assistant", "content": resp.text, "response": resp})
    ss.memory.add(Turn(question, resp.text, summarise_code(resp.artifacts), resp))


# --------------------------------------------------------------------- tabs
def tab_chat(mode, api_key, model, settings, pending: str | None) -> None:
    ss = st.session_state
    if not ss.store.tables:
        st.info("👈 Upload one or more CSV files, or click **Load sample dataset** to try the app.")
        return
    if not ss.messages:
        st.markdown("#### Try a question")
        cols = st.columns(4)
        for i, q in enumerate(SUGGESTIONS):
            if cols[i % 4].button(q, key=f"sugg-{i}", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()
    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant" and m.get("response"):
                render_response(m["response"], i)
            else:
                st.markdown(m["content"])
    if pending:
        handle_question(pending, mode, api_key, model, settings)


def tab_data() -> None:
    store = st.session_state.store
    if not store.tables:
        st.info("No data loaded yet.")
        return
    issues = store.quality_report()
    c1, c2, c3 = st.columns(3)
    c1.metric("Tables", len(store.tables))
    c2.metric("Total rows", f"{sum(t.rows for t in store.tables.values()):,}")
    c3.metric("Quality issues", len(issues))
    for t in store.tables.values():
        with st.expander(f"{t.name}  ·  {t.rows:,} rows  ·  {t.source}", expanded=False):
            st.dataframe(store.frames[t.name].head(100), width="stretch", hide_index=True)
            schema = pd.DataFrame([{"column": c.name, "type": c.kind, "nulls": c.nulls, "unique": c.unique,
                                    "min": c.min, "max": c.max} for c in t.columns]).astype(str)
            st.caption("Schema")
            st.dataframe(schema, hide_index=True, width="stretch")
    joins = store.suggest_joins()
    if joins:
        st.subheader("Detected relationships")
        st.dataframe(pd.DataFrame(joins).round({"match_pct": 1}), hide_index=True, width="stretch")
    st.subheader("Data quality")
    if issues:
        st.dataframe(pd.DataFrame(issues), hide_index=True, width="stretch")
    else:
        st.success("No data-quality issues detected.")


def tab_dashboard() -> None:
    store = st.session_state.store
    if not store.tables:
        st.info("Load data to generate a dashboard.")
        return
    table = st.selectbox("Table", list(store.tables), index=list(store.tables).index(store.require_table(None)))
    dash = build_dashboard(store, table)
    cols = st.columns(len(dash.kpis))
    for col, (label, value) in zip(cols, dash.kpis):
        col.metric(label, value)
    for i in range(0, len(dash.figures), 2):
        row = st.columns(2)
        for col, (title, fig) in zip(row, dash.figures[i:i + 2]):
            col.plotly_chart(fig, width="stretch", key=f"dash-{table}-{i}-{title}")


def tab_report() -> None:
    ss = st.session_state
    if not ss.memory.turns:
        st.info("Ask a few questions first; the report is built from your conversation.")
        return
    md, html_doc = build_markdown(ss.memory.turns, ss.store), build_html(ss.memory.turns, ss.store)
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ Download HTML report (with charts)", html_doc, "analysis_report.html", "text/html", width="stretch")
    c2.download_button("⬇️ Download Markdown report", md, "analysis_report.md", "text/markdown", width="stretch")
    st.markdown(md)


def main() -> None:
    init_state()
    if not password_gate():
        return
    settings = get_settings()
    mode, api_key, model = sidebar(settings)
    prompt = st.chat_input("Ask a question about your data...", disabled=not st.session_state.store.tables)
    pending = prompt or st.session_state.pop("pending_question", None)
    chat, data, dash, report = st.tabs(["💬 Chat", "🗂️ Data & quality", "📈 Dashboard", "📄 Report"])
    with chat:
        tab_chat(mode, api_key, model, settings, pending)
    with data:
        tab_data()
    with dash:
        tab_dashboard()
    with report:
        tab_report()


main()
