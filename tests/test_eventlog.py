"""The other shape a table can have, and deciding which shape a file is.

`TableSpec` maps COLUMN NAMES to activities — a tracker export, where a row is an
invoice and `approved_date` is a column. An event log is the opposite: the
activity is a VALUE in a cell and one case spans many rows. That is not a
formatting quibble. A wide table has exactly one `approved_date` column, so a
step can happen at most once, and a rework loop is unrepresentable in it — while
rework is the single most common finding a real log carries.

Long format is also the field's standard (the case / activity / timestamp
triple). So the engine has to read both, and — since a real file arrives without
a declaration — has to work out which it is holding and say what it measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from induction.adapters.tabular import (EventLogSpec, detect, load_event_log,
                                        parse_timestamp)

LOG = """\
Case ID;Event Name;Timestamp;Username;Vendor
INV-1;Receive;20/03/2024 2:10;bot_intake;Acme
INV-1;Verify;20/03/2024 4:25;Dana;Acme
INV-1;Verify;20/03/2024 6:26;Dana;Acme
INV-1;Approve;21/03/2024 9:00;Sam;Acme
INV-2;Receive;20/03/2024 3:10;bot_intake;Globex
INV-2;Verify;20/03/2024 5:25;Dana;Globex
INV-2;Approve;21/03/2024 8:00;Sam;Globex
INV-3;Receive;22/03/2024 3:10;bot_intake;Initech
INV-3;Verify;22/03/2024 5:25;Dana;Initech
INV-3;Approve;23/03/2024 8:00;Sam;Initech
INV-4;Receive;22/03/2024 3:40;bot_intake;Acme
INV-4;Verify;22/03/2024 5:45;Dana;Acme
INV-4;Approve;23/03/2024 8:30;Sam;Acme
INV-5;Receive;24/03/2024 3:10;bot_intake;Globex
INV-5;Verify;24/03/2024 5:25;Dana;Globex
INV-5;Approve;25/03/2024 8:00;Sam;Globex
INV-6;Receive;24/03/2024 4:10;bot_intake;Initech
INV-6;Verify;24/03/2024 6:25;Dana;Initech
INV-6;Approve;25/03/2024 9:00;Sam;Initech
"""


@pytest.fixture
def log_file(tmp_path) -> Path:
    p = tmp_path / "ap.csv"
    p.write_text(LOG)
    return p


def _spec():
    return EventLogSpec(source="log:acme/ap", entity_type="invoice",
                        case_id_column="Case ID", activity_column="Event Name",
                        timestamp_column="Timestamp", actor_column="Username",
                        attr_columns=["Vendor"])


# --- reading -----------------------------------------------------------------

def test_a_case_spans_many_rows(log_file):
    shaped = load_event_log(_spec(), log_file)
    invoices = [e for e in shaped.entities if e.type == "invoice"]
    assert [e.id for e in invoices] == [f"invoice:INV-{i}" for i in range(1, 7)]
    assert len(shaped.events) == 19                     # one event per row, not per case


def test_a_repeated_step_survives(log_file):
    """The reason long format exists. INV-1 verifies twice; a wide table has one
    `verify_date` column and would silently keep one of them."""
    shaped = load_event_log(_spec(), log_file)
    verifies = [e for e in shaped.events
                if e.entity_id == "invoice:INV-1" and e.action == "Verify"]
    assert len(verifies) == 2
    assert len({e.id for e in verifies}) == 2, "the two must be distinct events"
    assert {e.timestamp for e in verifies} == {"2024-03-20T04:25:00", "2024-03-20T06:26:00"}


def test_the_activity_is_the_sources_own_word(log_file):
    """Standing rule: an activity is named by the label the source gave it."""
    shaped = load_event_log(_spec(), log_file)
    assert {e.action for e in shaped.events} == {"Receive", "Verify", "Approve"}


def test_time_of_day_is_kept_so_a_case_can_be_ordered(log_file):
    """Several events land on one day; a date alone cannot order them."""
    assert parse_timestamp("20/03/2024 2:10") == "2024-03-20T02:10:00"
    assert parse_timestamp("") is None
    assert parse_timestamp("not a date") is None


def test_the_case_id_is_declared_as_a_link_not_assumed(log_file):
    """The one place a run needs no inference — but it still goes through the
    single correlator, like a merge DAG or a foreign key."""
    shaped = load_event_log(_spec(), log_file)
    ent = next(e for e in shaped.entities if e.id == "invoice:INV-1")
    from induction.links import ATTR
    links = ent.attrs.get(ATTR, [])
    assert links, "the case id must be declared, not silently assumed"
    assert any(l["method"] == "case-id" for l in links)
    assert all(l["tier"] == "direct" for l in links if l["method"] == "case-id")


def test_a_row_with_no_case_or_no_activity_is_surfaced(tmp_path):
    p = tmp_path / "gaps.csv"
    p.write_text(LOG + ";Approve;24/03/2024 8:00;Sam;Acme\nINV-9;;24/03/2024 9:00;Sam;Acme\n")
    shaped = load_event_log(_spec(), p)
    orphans = [e for e in shaped.entities if e.type == "orphan_row"]
    assert {e.attrs["reason"] for e in orphans} == {"no case id", "no activity"}
    assert all(e.evidence for e in orphans), "a surfaced row must say where it came from"


def test_max_cases_caps_the_slice(log_file):
    shaped = load_event_log(_spec(), log_file, max_cases=2)
    assert len([e for e in shaped.entities if e.type == "invoice"]) == 2


# --- detection ---------------------------------------------------------------

def test_an_event_log_is_recognised_and_the_triple_named(log_file):
    got = detect(log_file, entity_type="invoice")
    assert got.mode == "long"
    assert (got.spec.case_id_column, got.spec.activity_column,
            got.spec.timestamp_column) == ("Case ID", "Event Name", "Timestamp")
    assert got.spec.actor_column == "Username"
    assert got.confidence.tier.label == "heuristic", "a detected shape is inferred, not read"
    assert "Event Name" in got.rationale and "Case ID" in got.rationale


def test_detection_measures_variation_inside_a_case_not_column_size(log_file):
    """The first attempt scored raw 'spread' and picked a 3-value attribute as
    the case id, because with three huge groups every column varies a lot inside
    one. What separates an activity from a status is the FRACTION of a case's
    rows carrying a new value — scale-free, and the reason 'Vendor' loses here."""
    got = detect(log_file, entity_type="invoice")
    assert got.spec.case_id_column != "Vendor"
    assert got.spec.activity_column != "Vendor"


def test_a_tracker_export_is_recognised_as_wide():
    got = detect("samples/finance/invoices.csv", entity_type="invoice")
    assert got.mode == "wide"
    assert got.spec.id_column == "invoice_id"
    assert [e.action for e in got.spec.event_columns] == [
        "raised", "submitted", "approved", "paid"]


def test_detection_reproduces_the_hand_written_spec():
    """The strongest check available: on corpora whose mapping a human wrote by
    hand, detection must arrive at the same answer, actor columns included."""
    import run_tabular
    hand = {e.date_column: e.actor_column
            for e in run_tabular.sources_for(Path("samples/finance"), "csv")[0][0].event_columns}
    found = {e.date_column: e.actor_column
             for e in detect("samples/finance/invoices.csv").spec.event_columns}
    assert found == hand


def test_a_messy_identity_column_is_still_an_identity():
    """samples/finance has duplicate and blank invoice ids on purpose — those are
    findings, not grounds to reject the file. Demanding a perfect key would throw
    out exactly the messy sheet this engine is for."""
    got = detect("samples/finance/invoices.csv", entity_type="invoice")
    assert got.mode == "wide" and got.spec.id_column == "invoice_id"


def test_a_table_that_is_neither_says_so():
    got = detect("samples/finance/payments.csv", entity_type="payment")
    assert got.mode is None
    assert "neither shape fits" in got.rationale


def test_a_semicolon_export_is_read_as_columns_not_one_blob(log_file):
    """A European export is `;`-delimited; read with a comma every row becomes a
    single unusable column and every later check fails for the wrong reason."""
    from induction.adapters.tabular import read_rows
    rows = read_rows(log_file)
    assert rows[0]["Case ID"] == "INV-1"
    assert rows[0]["Event Name"] == "Receive"
