"""Schedule integrity, and the provenance rule the whole repo depends on."""

from __future__ import annotations

from datetime import date

import pytest
import yaml

from aurex.config import SCHEDULE_DIR
from aurex.data.schedules import (
    VALID_CONFIDENCE,
    ScheduleError,
    _require_provenance,
    duty_on,
    gst_on,
    load_duty_schedule,
    load_gst_schedule,
    load_policy_breaks,
)


class TestProvenanceRule:
    """Every entry cites its own source at its own confidence. No inheritance."""

    @pytest.mark.parametrize("filename", ["duty.yaml", "gst.yaml"])
    def test_every_entry_has_its_own_provenance(self, filename: str) -> None:
        raw = yaml.safe_load((SCHEDULE_DIR / filename).read_text())
        entries = raw["schedule"]
        assert entries, f"{filename} has no entries"

        for i, entry in enumerate(entries):
            assert entry.get("source_url"), f"{filename}[{i}] missing source_url"
            assert entry.get("source_confidence"), f"{filename}[{i}] missing source_confidence"
            assert entry["source_confidence"] in VALID_CONFIDENCE

    def test_policy_breaks_are_all_sourced(self) -> None:
        for brk in load_policy_breaks():
            assert brk.source_url.startswith("http"), f"{brk.date} has no usable source_url"

    def test_missing_source_url_is_rejected(self) -> None:
        with pytest.raises(ScheduleError, match="missing source_url"):
            _require_provenance("x.yaml", 0, {"source_confidence": "primary"})

    def test_missing_confidence_is_rejected(self) -> None:
        with pytest.raises(ScheduleError, match="missing source_confidence"):
            _require_provenance("x.yaml", 0, {"source_url": "https://example.invalid"})

    def test_unknown_confidence_is_rejected(self) -> None:
        with pytest.raises(ScheduleError, match="not in"):
            _require_provenance(
                "x.yaml", 0, {"source_url": "https://e.invalid", "source_confidence": "vibes"}
            )


class TestDutySchedule:
    def test_sorted_ascending(self) -> None:
        entries = load_duty_schedule()
        dates = [e.effective_from for e in entries]
        assert dates == sorted(dates)

    def test_components_sum_to_total(self) -> None:
        """`total` must equal what the components say, or the table lies."""
        for entry in load_duty_schedule():
            summed = sum(entry.components.values())
            assert summed == pytest.approx(entry.total, abs=1e-9), (
                f"{entry.effective_from}: components {entry.components} "
                f"sum to {summed} but total says {entry.total}"
            )

    def test_current_duty_is_fifteen_percent(self) -> None:
        """Effective 2026-05-13: BCD 10% + AIDC 5% + SWS nil."""
        entry = duty_on(date(2026, 7, 29))
        assert entry is not None
        assert entry.total == pytest.approx(0.15)
        assert entry.components["bcd"] == pytest.approx(0.10)
        assert entry.components["aidc"] == pytest.approx(0.05)
        assert entry.components["sws"] == pytest.approx(0.0)

    def test_sws_is_nil_on_bullion_throughout(self) -> None:
        """The 15%-vs-16% confusion comes from charging gold an SWS it never pays."""
        for entry in load_duty_schedule():
            assert entry.components.get("sws", 0.0) == pytest.approx(0.0)

    def test_duty_resolves_to_the_entry_in_force(self) -> None:
        assert duty_on(date(2026, 5, 12)).total == pytest.approx(0.06)  # type: ignore[union-attr]
        assert duty_on(date(2026, 5, 13)).total == pytest.approx(0.15)  # type: ignore[union-attr]
        assert duty_on(date(2024, 7, 22)).total == pytest.approx(0.15)  # type: ignore[union-attr]
        assert duty_on(date(2024, 7, 23)).total == pytest.approx(0.06)  # type: ignore[union-attr]

    def test_before_ad_valorem_regime_there_is_no_rate(self) -> None:
        """Pre-2012 duty was Rs 300/10g specific. A percentage would be invented."""
        assert duty_on(date(2011, 12, 31)) is None


class TestGstSchedule:
    def test_metal_rate_is_three_percent(self) -> None:
        entry = gst_on(date(2026, 7, 29))
        assert entry is not None
        assert entry.metal == pytest.approx(0.03)
        assert entry.making_charges == pytest.approx(0.05)

    def test_no_gst_before_rollout(self) -> None:
        assert gst_on(date(2017, 6, 30)) is None
        assert gst_on(date(2017, 7, 1)) is not None

    def test_sorted_ascending(self) -> None:
        dates = [e.effective_from for e in load_gst_schedule()]
        assert dates == sorted(dates)


class TestPolicyBreaks:
    def test_duty_revisions_appear_as_breaks(self) -> None:
        """Every duty change must be recorded as a break, or downstream layers
        will treat a mechanical step as a volatility shock."""
        break_dates = {b.date for b in load_policy_breaks()}
        duty_dates = {e.effective_from for e in load_duty_schedule()}
        missing = duty_dates - break_dates

        # 2012-03/2013-01/2013-06 are consecutive small steps inside one regime and
        # are represented by the 2012-01-17 basis change and the 2013-08-13 entry.
        allowed_gaps = {date(2012, 3, 17), date(2013, 1, 21), date(2013, 6, 5)}
        assert missing <= allowed_gaps, f"unrecorded duty breaks: {sorted(missing - allowed_gaps)}"

    def test_the_2026_break_is_present(self) -> None:
        assert date(2026, 5, 13) in {b.date for b in load_policy_breaks()}

    def test_sorted_ascending(self) -> None:
        dates = [b.date for b in load_policy_breaks()]
        assert dates == sorted(dates)
