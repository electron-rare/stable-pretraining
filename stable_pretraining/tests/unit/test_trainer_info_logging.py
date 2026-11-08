"""Logged tables contain real newlines and no ANSI codes outside a TTY."""

import io
from types import SimpleNamespace

import pytest
import torch
from loguru import logger
from torch import nn

from stable_pretraining.callbacks import trainer_info
from stable_pretraining.callbacks.trainer_info import LoggingCallback, ModuleSummary

pytestmark = pytest.mark.unit


class _FakeStdout(io.StringIO):
    def __init__(self, tty):
        super().__init__()
        self._tty = tty

    def isatty(self):
        return self._tty


def _metrics_trainer():
    return SimpleNamespace(
        callback_metrics={
            "fit/loss": torch.tensor(0.5),
            "val/acc": torch.tensor(0.75),
        }
    )


@pytest.fixture
def captured_log():
    """Collect formatted log records in a non-TTY sink."""
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", colorize=False)
    yield sink
    logger.remove(handler_id)


def test_tables_have_no_escapes_when_not_a_tty(monkeypatch, captured_log):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(trainer_info.sys, "stdout", _FakeStdout(tty=False))

    ModuleSummary().setup(None, nn.Sequential(nn.Linear(4, 2)), "fit")
    LoggingCallback().on_validation_end(_metrics_trainer(), None)

    text = captured_log.getvalue()
    assert "\x1b" not in text
    assert "\\n" not in text
    assert "\n+--------+" in text
    assert "| fit/loss |  0.5  |" in text


def test_metric_table_colored_on_tty(monkeypatch, captured_log):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(trainer_info.sys, "stdout", _FakeStdout(tty=True))

    LoggingCallback().on_validation_end(_metrics_trainer(), None)

    assert "\x1b[0;34;40mfit/loss\x1b[0m" in captured_log.getvalue()


def test_no_color_disables_colors_on_tty(monkeypatch, captured_log):
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.setattr(trainer_info.sys, "stdout", _FakeStdout(tty=True))

    LoggingCallback().on_validation_end(_metrics_trainer(), None)

    assert "\x1b" not in captured_log.getvalue()
