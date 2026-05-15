from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TextIO

import sys


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    current: int = 0
    total: int | None = None
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


class ConsoleProgress:
    def __init__(self, enabled: bool = True, stream: TextIO | None = None):
        self.enabled = enabled
        self.stream = stream or sys.stderr
        self._last_stage: str | None = None
        self._line_open = False

    def __call__(self, event: ProgressEvent) -> None:
        if not self.enabled:
            return
        if self._last_stage != event.stage and self._line_open:
            self.stream.write("\n")
            self._line_open = False
        self._last_stage = event.stage

        if event.total and event.total > 0:
            pct = min(max(event.current / event.total, 0.0), 1.0)
            width = 28
            filled = int(width * pct)
            bar = "#" * filled + "-" * (width - filled)
            suffix = f" {event.message}" if event.message else ""
            self.stream.write(
                f"\r{event.stage:<18} [{bar}] {event.current}/{event.total}{suffix}"
            )
            self._line_open = True
            if event.current >= event.total:
                self.stream.write("\n")
                self._line_open = False
        else:
            if self._line_open:
                self.stream.write("\n")
                self._line_open = False
            suffix = f": {event.message}" if event.message else ""
            self.stream.write(f"{event.stage}{suffix}\n")
        self.stream.flush()
