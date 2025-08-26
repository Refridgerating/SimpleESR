import sys
import pytest

from esr_lab import gui

def test_has_display(monkeypatch):
    if sys.platform.startswith("win"):
        pytest.skip("DISPLAY not used on Windows")
    monkeypatch.delenv("DISPLAY", raising=False)
    assert gui.has_display() is False
    monkeypatch.setenv("DISPLAY", ":0")
    assert gui.has_display() is True
