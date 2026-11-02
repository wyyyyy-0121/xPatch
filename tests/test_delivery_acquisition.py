"""Provenance, recovery and authentication boundaries for new acquisition."""
import copy
import json

import pytest

from data_acquisition import delivery_data as acquisition


def bar(stamp, close=100):
    return {"t": stamp, "o": close, "h": close + 1, "l": close - 1,
            "c": close, "v": 100, "n": 3, "vw": float(close)}


def payload(rows, token=None):
    return json.dumps({"bars": {"SPY": rows}, "next_page_token": token}).encode()


@pytest.fixture
def acquired(tmp_path, monkeypatch):
    class FakeClient:
        def get(self, path, params):
            assert path == "/v2/stocks/bars"
            assert params["feed"] == "sip"
            assert params["adjustment"] == "raw"
            if "page_token" not in params:
                return 200, payload([bar("2024-03-11T13:30:00Z")], "next"), {}
            assert params["page_token"] == "next"
            return 200, payload([bar("2024-03-11T13:31:00Z", 101)]), {}
    monkeypatch.setattr(acquisition, "Client", FakeClient)
    root = tmp_path / "run"
    acquisition.download("SPY", [("2024-03-11", "2024-03-11")], root)
    return root


def test_all_fields_and_dst_are_preserved(acquired):
    result = acquisition.verify(acquired)
    assert result["rows"] == 2
    assert result["pages"] == 2
    assert result["all_fields_verified"] == acquisition.COLUMNS
    text = (acquired / "SPY_sip_1min_raw.csv").read_text()
    assert "2024-03-11T09:30:00-04:00" in text
    assert result["point_in_time_available"] is False


@pytest.mark.parametrize("kind", ["raw", "csv", "pagination", "receipt_identity"])
def test_tampering_is_rejected(acquired, kind):
    receipt_path = acquired / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if kind == "raw":
        path = acquired / receipt["pages"][0]["file"]
        path.write_text(path.read_text().replace('"c": 100', '"c": 102'))
    elif kind == "csv":
        path = acquired / receipt["csv"]
        path.write_text(path.read_text().replace("101,102", "108,102"))
    elif kind == "pagination":
        receipt["pages"] = receipt["pages"][:1]
        receipt_path.write_text(json.dumps(receipt))
    else:
        receipt["request"]["feed"] = "iex"
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        acquisition.verify(acquired)


def test_incomplete_download_recovers_without_refetching_cached_page(tmp_path, monkeypatch):
    calls = []
    fail = [True]
    class FakeClient:
        def get(self, path, params):
            calls.append(copy.deepcopy(params))
            if "page_token" not in params:
                return 200, payload([bar("2024-03-11T13:30:00Z")], "second"), {}
            if fail[0]:
                raise RuntimeError("simulated network failure")
            return 200, payload([bar("2024-03-11T13:31:00Z")]), {}
    monkeypatch.setattr(acquisition, "Client", FakeClient)
    root = tmp_path / "recovery"
    ranges = [("2024-03-11", "2024-03-11")]
    with pytest.raises(RuntimeError):
        acquisition.download("SPY", ranges, root)
    assert json.loads((root / "receipt.json").read_text())["download_complete"] is False
    fail[0] = False
    assert acquisition.download("SPY", ranges, root)["passed"]
    assert len(calls) == 3
    assert calls[-1]["page_token"] == "second"
    with pytest.raises(ValueError, match="identity"):
        acquisition.download("QQQ", ranges, root)


def test_authentication_has_no_redirect_or_nonmarket_route(monkeypatch):
    monkeypatch.setenv("APCA_API_KEY_ID", "test-placeholder")
    monkeypatch.setenv("APCA_API_SECRET_KEY", "test-placeholder")
    client = acquisition.Client()
    for path in ("/v2/orders", "https://example.invalid/", "/v2/stocks/../orders", "/v2/stocks/bars?other"):
        with pytest.raises(ValueError, match="allowlist"):
            client.get(path, {})
    assert acquisition.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.invalid") is None


def test_credential_file_is_never_executed(tmp_path, monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    monkeypatch.setattr(acquisition.Path, "home", lambda: tmp_path)
    config = tmp_path / ".config/alpaca"
    config.mkdir(parents=True)
    marker = tmp_path / "must_not_exist"
    (config / "credentials.env").write_text(
        f"APCA_API_KEY_ID='$(touch {marker})'\nAPCA_API_SECRET_KEY=placeholder\n")
    acquisition.load_environment()
    assert not marker.exists()
    assert acquisition.os.environ["APCA_API_KEY_ID"].startswith("$(touch ")
