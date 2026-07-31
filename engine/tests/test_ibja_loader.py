"""The IBJA fetch loop: weekend skipping, request budgeting, history accumulation.

Networking is stubbed at ``_fetch_one`` so the loop logic is tested in isolation.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from aurex.data.sources.ibja import IbjaReportLoader, parse_report


def report(day: str, pm: float, tonnes: float = 1008.73, history: list | None = None) -> dict:
    return {
        "report_date": pd.Timestamp(day),
        "gold_999_am": pm + 245.0,
        "gold_999_pm": pm,
        "spdr_gold_tonnes": tonnes,
        "london_pm_usd": 4000.85,
        "spot_history": history or [],
    }


class TestFetchLoop:
    def test_builds_a_frame_from_one_report(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            IbjaReportLoader, "_fetch_one", lambda self, d: report("2026-07-30", 142224.0)
        )
        loaded = IbjaReportLoader(max_reports=1).fetch(date(2026, 7, 1), date(2026, 7, 30))

        assert loaded.frame.loc[pd.Timestamp("2026-07-30"), "gold_999_pm"] == 142224.0
        assert loaded.frame.loc[pd.Timestamp("2026-07-30"), "spdr_gold_tonnes"] == 1008.73
        assert loaded.meta.source_name == "IBJA:daily-bullion-report"

    def test_embedded_history_extends_the_series(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """One report yields several days, which is why the loader need not crawl."""
        history = [
            (pd.Timestamp("2026-07-29"), 142224.0),
            (pd.Timestamp("2026-07-28"), 142291.0),
            (pd.Timestamp("2026-07-27"), 144466.0),
        ]
        monkeypatch.setattr(
            IbjaReportLoader,
            "_fetch_one",
            lambda self, d: report("2026-07-30", 142224.0, history=history),
        )
        loaded = IbjaReportLoader(max_reports=1).fetch(date(2026, 7, 1), date(2026, 7, 30))

        assert len(loaded.frame) == 4
        assert loaded.frame.loc[pd.Timestamp("2026-07-27"), "gold_999_pm"] == 144466.0

    def test_report_values_win_over_embedded_history(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The header block is authoritative for its own date."""
        history = [(pd.Timestamp("2026-07-30"), 999.0)]
        monkeypatch.setattr(
            IbjaReportLoader,
            "_fetch_one",
            lambda self, d: report("2026-07-30", 142224.0, history=history),
        )
        loaded = IbjaReportLoader(max_reports=1).fetch(date(2026, 7, 1), date(2026, 7, 30))
        assert loaded.frame.loc[pd.Timestamp("2026-07-30"), "gold_999_pm"] == 142224.0

    def test_weekends_are_skipped_without_spending_requests(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """IBJA does not publish on weekends; requesting anyway is rude and useless."""
        asked: list[date] = []

        def record(self: Any, day: date) -> dict | None:
            asked.append(day)
            return report(day.isoformat(), 142000.0)

        monkeypatch.setattr(IbjaReportLoader, "_fetch_one", record)
        # 2026-08-01 is a Saturday, 2026-08-02 a Sunday.
        IbjaReportLoader(max_reports=2).fetch(date(2026, 7, 20), date(2026, 8, 2))

        assert all(d.weekday() < 5 for d in asked), f"weekend requests made: {asked}"

    def test_request_budget_is_respected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[date] = []

        def record(self: Any, day: date) -> dict | None:
            calls.append(day)
            return report(day.isoformat(), 142000.0)

        monkeypatch.setattr(IbjaReportLoader, "_fetch_one", record)
        IbjaReportLoader(max_reports=3).fetch(date(2026, 1, 1), date(2026, 7, 30))
        assert len(calls) == 3

    def test_unpublished_days_are_tolerated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A holiday returns nothing; the loop keeps walking back."""
        seen: list[date] = []

        def sometimes(self: Any, day: date) -> dict | None:
            seen.append(day)
            return None if len(seen) < 3 else report(day.isoformat(), 142000.0)

        monkeypatch.setattr(IbjaReportLoader, "_fetch_one", sometimes)
        loaded = IbjaReportLoader(max_reports=5).fetch(date(2026, 7, 1), date(2026, 7, 30))
        assert not loaded.frame.empty

    def test_no_reports_raises_so_the_chain_can_fall_through(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(IbjaReportLoader, "_fetch_one", lambda self, d: None)
        with pytest.raises(ValueError, match="no reports parsed"):
            IbjaReportLoader(max_reports=2).fetch(date(2026, 7, 1), date(2026, 7, 30))

    def test_rows_outside_the_window_are_excluded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        history = [(pd.Timestamp("2020-01-02"), 40000.0)]
        monkeypatch.setattr(
            IbjaReportLoader,
            "_fetch_one",
            lambda self, d: report("2026-07-30", 142224.0, history=history),
        )
        loaded = IbjaReportLoader(max_reports=1).fetch(date(2026, 7, 1), date(2026, 7, 30))
        assert pd.Timestamp("2020-01-02") not in loaded.frame.index


class TestParserRobustness:
    def test_a_malformed_date_does_not_crash_the_parser(self) -> None:
        """IBJA has published typo'd dates; drop the row, keep the report."""
        text = "Date: 30th Julyy 2026\nGold 999 142469 142224\n"
        parsed = parse_report(text)
        assert parsed["report_date"] is None
        assert parsed["gold_999_pm"] == 142224.0

    def test_impossible_calendar_date_is_dropped(self) -> None:
        parsed = parse_report("31st February 2026 142224 217335")
        assert parsed["spot_history"] == []
