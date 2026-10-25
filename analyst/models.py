"""Shared data containers passed between the agent, tools and the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class TurnArtifacts:
    """Everything produced while answering one question (shown in the UI)."""

    steps: list[str] = field(default_factory=list)      # human-readable reasoning trace
    code: list[dict] = field(default_factory=list)      # {"language","code","purpose","rows"}
    tables: list[dict] = field(default_factory=list)    # {"title","df"}
    figures: list[dict] = field(default_factory=list)   # {"title","fig"}

    def add_step(self, text: str) -> None:
        self.steps.append(text)

    def add_code(self, language: str, code: str, purpose: str = "", rows: int | None = None) -> None:
        self.code.append({"language": language, "code": code, "purpose": purpose, "rows": rows})

    def add_table(self, title: str, df: pd.DataFrame) -> None:
        self.tables.append({"title": title, "df": df})

    def add_figure(self, title: str, fig: Any) -> None:
        self.figures.append({"title": title, "fig": fig})


@dataclass
class AgentResponse:
    text: str
    artifacts: TurnArtifacts
    mode: str = "offline"                 # "llm" or "offline"
    latency_s: float = 0.0
    usage: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class Turn:
    """One completed question/answer exchange kept in session memory."""

    question: str
    answer: str
    code_summary: str = ""
    response: AgentResponse | None = None
