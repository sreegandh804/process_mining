"""The tabular (spreadsheet) adapter + generic correlate/gaps — the non-git
customer, end to end. Small hand-verified fixture; exact, honest assertions.
"""

import importlib

import pytest

from induction.adapters.tabular import EventCol, TableSpec, parse_date
from induction.pipeline import run_tabular_pipeline
from induction.profiles import ACCOUNTING_PROFILE, GENERIC_PROFILE

INVOICES = """\
inv,client,status,raised_date,raised_by,approved_date,approved_by,paid_date
I-1,Acme,Paid,2024-01-01,Sam,2024-01-03,Mia,2024-01-10
I-2,Beta,Paid,2024-01-05,Sam,,,2024-01-12
I-3,Gamma,Draft,,,,,
I-1,Acme,Paid,2024-01-01,Sam,2024-01-03,Mia,2024-01-11
,Ghost,Paid,2024-01-02,Sam,,,2024-01-09
I-4,Delta,Paid,02/01/2024,Sam,04/01/2024,Mia,10/01/2024
I-5,Sys,Paid,2024-01-01,system,2024-01-01,system,2024-01-01
I-6,Sys,Paid,2024-02-01,system,2024-02-01,system,2024-02-01
"""
PAYMENTS = """\
ref,inv,paid_date
P-1,I-1,2024-01-10
P-2,I-99,2024-01-05
"""


def _sources(dir_):
    (dir_ / "invoices.csv").write_text(INVOICES)
    (dir_ / "payments.csv").write_text(PAYMENTS)
    inv = TableSpec(
        source="sheet:test/invoices", entity_type="invoice", id_column="inv",
        event_columns=[EventCol("raised", "raised_date", "raised_by"),
                       EventCol("approved", "approved_date", "approved_by"),
                       EventCol("paid", "paid_date")],
        status_column="status", attr_columns=["client"])
    pay = TableSpec(
        source="sheet:test/payments", entity_type="payment", id_column="ref",
        event_columns=[EventCol("settled", "paid_date")],
        ref_columns=[{"column": "inv", "target_type": "invoice"}])
    return [(inv, dir_ / "invoices.csv"), (pay, dir_ / "payments.csv")]


@pytest.fixture(scope="module")
def fin(tmp_path_factory):
    d = tmp_path_factory.mktemp("fin")
    return run_tabular_pipeline(_sources(d), slug="test-fin")


def test_dates_parse_across_formats():
    assert parse_date("2024-01-02") == "2024-01-02"
    assert parse_date("02/01/2024") == "2024-01-02"   # day-first
    assert parse_date("2-Jan-2024") == "2024-01-02"
    assert parse_date("") is None
    assert parse_date("not a date") is None            # unparseable -> None, not a guess


def test_each_invoice_is_a_case_built_from_its_row(fin):
    case = fin.cases["case:invoice:I-1"]
    actions = {e.action for e in fin.shaped.events if e.case_id == case.id}
    assert {"raised", "approved", "paid"} <= actions


def test_cross_source_join_payment_settles_the_invoice(fin):
    # P-1 references I-1 -> its 'settled' event joins the invoice's run
    actions = {e.action for e in fin.shaped.events if e.case_id == "case:invoice:I-1"}
    assert "settled" in actions


def test_missing_step_gap_is_paid_without_approval(fin):
    miss = [g for g in fin.gaps if g.kind == "missing_expected_step"]
    assert any(g.case_id == "case:invoice:I-2" and "approved" in g.description for g in miss)
    for g in miss:                       # always inference, never fact
        assert g.confidence.tier.label in {"heuristic", "model"}


def test_reconciliation_finds_unmatched_records(fin):
    rec = [g for g in fin.gaps if g.kind == "reconciliation"]
    assert any("I-99" in g.description for g in rec)          # payment -> unknown invoice
    assert any(g.case_id == "case:invoice:I-2" for g in rec)  # paid, no bank settlement


def test_blank_actor_and_no_date_are_never_invented(fin):
    # I-2's paid event has no actor column value -> actor None
    paid = next(e for e in fin.shaped.events
                if e.case_id == "case:invoice:I-2" and e.action == "paid")
    assert paid.actor is None
    # I-3 is Draft with no dates -> order unknown, from a state Observation
    assert fin.cases["case:invoice:I-3"].order_status == "unknown"


def test_row_with_no_identity_becomes_an_orphan(fin):
    assert any(o.record_type == "observation" for o in fin.orphans)
    assert fin.orphans, "the id-less Ghost row must be surfaced, not dropped"


def test_duplicate_identity_is_flagged(fin):
    inv1 = next(e for e in fin.shaped.entities if e.id == "invoice:I-1")
    assert inv1.attrs.get("duplicate_id") is True


def test_recurring_system_actor_cluster_is_rejected(fin):
    assert any(k.rejected for k in fin.kinds)


def test_generic_default_is_unnamed_accounting_profile_names(tmp_path):
    src = _sources(tmp_path)
    generic = run_tabular_pipeline(src, slug="t", profile=GENERIC_PROFILE)
    named = run_tabular_pipeline(src, slug="t", profile=ACCOUNTING_PROFILE)
    assert all(k.id.startswith("kind_") for k in generic.kinds)
    assert any(k.name == "Invoice approval & payment" for k in named.kinds)
    # vocabulary only — identical structure
    assert set(generic.cases) == set(named.cases)
    assert len(generic.gaps) == len(named.gaps)


def test_xlsx_matches_csv_when_openpyxl_present(tmp_path):
    if importlib.util.find_spec("openpyxl") is None:
        pytest.skip("openpyxl not installed; the .xlsx path is optional")
    import openpyxl
    src = _sources(tmp_path)
    # write matching .xlsx sheets and re-point the specs
    xlsx_sources = []
    for spec, csv_path in src:
        wb = openpyxl.Workbook(); ws = wb.active
        for line in csv_path.read_text().splitlines():
            ws.append(line.split(","))
        xpath = csv_path.with_suffix(".xlsx"); wb.save(xpath)
        xlsx_sources.append((spec, xpath))
    csv_m = run_tabular_pipeline(src, slug="t")
    xlsx_m = run_tabular_pipeline(xlsx_sources, slug="t")
    assert set(csv_m.cases) == set(xlsx_m.cases)
    assert len(csv_m.gaps) == len(xlsx_m.gaps)
