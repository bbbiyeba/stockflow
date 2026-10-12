import pytest

from src import access
from src import decisions as dlog


def test_record_and_latest_roundtrip(tmp_path):
    p = tmp_path / "d.csv"
    n = dlog.record(
        p,
        "run-1",
        [
            {"transfer_id": "abc123", "decision": "approved", "note": "ok"},
            {"transfer_id": "def456", "decision": "rejected"},
        ],
        "Pat",
        "approver",
    )
    assert n == 2
    dlog.record(
        p, "run-1", [{"transfer_id": "abc123", "decision": "rejected", "note": "changed mind"}], "Sam", "approver"
    )
    latest = dlog.latest(p, "run-1").set_index("transfer_id")
    assert latest.loc["abc123", "decision"] == "rejected"
    assert latest.loc["abc123", "reviewer"] == "Sam"
    assert len(dlog.history(p)) == 3  # append-only


def test_viewers_cannot_record(tmp_path):
    with pytest.raises(dlog.DecisionError):
        dlog.record(tmp_path / "d.csv", "run-1", [{"transfer_id": "a", "decision": "approved"}], "Pat", "viewer")


@pytest.mark.parametrize(
    "row", [{"transfer_id": "a;rm -rf", "decision": "approved"}, {"transfer_id": "abc", "decision": "maybe"}]
)
def test_invalid_rows_rejected(tmp_path, row):
    with pytest.raises(dlog.DecisionError):
        dlog.record(tmp_path / "d.csv", "run-1", [row], "Pat", "approver")


def test_notes_are_single_line_and_formula_safe(tmp_path):
    p = tmp_path / "d.csv"
    dlog.record(
        p,
        "run-1",
        [{"transfer_id": "a1", "decision": "approved", "note": '=HYPERLINK("x")\nline2'}],
        "@admin",
        "approver",
    )
    row = dlog.history(p).iloc[0]
    assert row["note"].startswith("'=") and "\n" not in row["note"]
    assert row["reviewer"].startswith("'@")


def test_empty_history(tmp_path):
    assert dlog.latest(tmp_path / "missing.csv").empty


def test_roles(monkeypatch):
    monkeypatch.delenv("STOCKFLOW_REQUIRE_LOGIN", raising=False)
    monkeypatch.delenv("STOCKFLOW_DEMO_APPROVALS", raising=False)
    assert access.resolve(None).role == "viewer"
    monkeypatch.setenv("STOCKFLOW_DEMO_APPROVALS", "1")
    assert access.resolve(None).role == "approver"
    monkeypatch.setenv("STOCKFLOW_APPROVERS", "boss@corp.com")
    signed_in = {"is_logged_in": True, "email": "Boss@corp.com", "name": "Boss"}
    assert access.resolve(signed_in).role == "approver"
    assert access.resolve({**signed_in, "email": "intern@corp.com"}).role == "viewer"
    monkeypatch.setenv("STOCKFLOW_REQUIRE_LOGIN", "1")
    u = access.resolve(None)
    assert not u.authenticated and u.role == "viewer"
