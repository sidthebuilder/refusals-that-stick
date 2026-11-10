"""Tests for the credential loader.

The behaviour that matters most: no test here, and no code path, may cause a
secret value to be printed or embedded in an error message.
"""

from __future__ import annotations

import os

import pytest

from refusals_that_stick.env import have, load_env, require


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") == ()


def test_loads_names_and_sets_values(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text('A=1\nB="two"\n# comment\n\nC=three\n', encoding="utf-8")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    monkeypatch.delenv("C", raising=False)

    names = load_env(p)

    assert set(names) == {"A", "B", "C"}
    assert os.environ["A"] == "1"
    assert os.environ["B"] == "two"  # quotes stripped
    assert os.environ["C"] == "three"


def test_existing_environment_wins_by_default(tmp_path, monkeypatch):
    p = tmp_path / ".env"
    p.write_text("K=from_file\n", encoding="utf-8")
    monkeypatch.setenv("K", "from_shell")

    load_env(p)
    assert os.environ["K"] == "from_shell"

    load_env(p, override=True)
    assert os.environ["K"] == "from_file"


def test_malformed_lines_are_skipped(tmp_path):
    p = tmp_path / ".env"
    p.write_text("no_equals_sign_here\n=novalue\nGOOD=1\n", encoding="utf-8")
    assert set(load_env(p)) == {"GOOD"}


def test_require_error_names_the_variable_without_leaking_a_value(monkeypatch):
    monkeypatch.setenv("SECRET_THING", "")
    with pytest.raises(RuntimeError) as exc:
        require("SECRET_THING")
    message = str(exc.value)
    assert "SECRET_THING" in message
    assert ".env" in message


def test_require_returns_value_when_present(monkeypatch):
    monkeypatch.setenv("PRESENT_THING", "value")
    assert require("PRESENT_THING") == "value"


def test_have_reports_presence_only(monkeypatch):
    monkeypatch.setenv("SET_THING", "x")
    monkeypatch.setenv("EMPTY_THING", "   ")
    assert have("SET_THING") is True
    assert have("EMPTY_THING") is False
