"""LLAMEA-style error memory: record runtime failures and inject them
back into future LLM prompts as an "avoid these patterns" hint.

Why this module exists
----------------------
Without this, when an LLM emits e.g. ``bins.sum(axis=1)`` on a 1-D array
the sandbox raises ``axis 1 is out of bounds``, the sample is dropped,
and the *next* LLM call has zero awareness — so it cheerfully repeats
the same mistake. With ~40% runtime-failure rate that is a lot of
wasted budget. We give the search a tiny working memory of the most
recent failures and append a short "do not do this" block to every
prompt; one failure now costs at most ``capacity`` budget instead of
infinity.

Design choices (kept deliberately small to avoid prompt bloat)
--------------------------------------------------------------
1. Bounded deque (capacity=5 by default). Old failures fall off
   automatically — we want the freshest signal, not a full audit log.
2. Each error message is truncated to ``max_msg_chars`` (default 160).
   Most useful traceback signal lives in the exception type and the
   first line of the message; the rest is noise.
3. Only *runtime* errors are recorded by the caller — structural
   ``InvalidSampleError`` (empty body, no return) is filtered upstream
   in ``sampler.py`` because telling the LLM "your body was empty"
   doesn't teach it anything strategic.
4. ``render_for_prompt`` returns ``''`` when empty so callers can
   unconditionally inject without having to guard.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ErrorRecord:
    """A single recorded failure observation."""
    op: str        # the EoH operator that produced the failing sample
    msg: str       # truncated, single-line error message


class ErrorMemory:
    """Bounded FIFO of recent runtime errors, renderable as a prompt block."""

    DEFAULT_CAPACITY = 5
    DEFAULT_MAX_MSG_CHARS = 160
    HEADER = '## Recent failures to avoid (do not repeat these mistakes):'

    def __init__(
            self,
            capacity: int = DEFAULT_CAPACITY,
            max_msg_chars: int = DEFAULT_MAX_MSG_CHARS,
    ):
        if capacity <= 0:
            raise ValueError(f'capacity must be > 0, got {capacity}')
        if max_msg_chars <= 0:
            raise ValueError(
                f'max_msg_chars must be > 0, got {max_msg_chars}')
        self._records: deque[ErrorRecord] = deque(maxlen=capacity)
        self._max_msg_chars = max_msg_chars
        self._total_recorded = 0

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------
    def record(self, op: str, error_msg: str | None) -> None:
        """Store one runtime failure. No-op if ``error_msg`` is empty.

        Multi-line messages are collapsed to their first non-empty line,
        then truncated to ``max_msg_chars``. This keeps the eventual
        prompt block under tight control (worst case
        ``capacity * max_msg_chars`` chars + header).
        """
        if not error_msg:
            return
        first_line = next(
            (ln.strip() for ln in str(error_msg).splitlines() if ln.strip()),
            '',
        )
        if not first_line:
            return
        if len(first_line) > self._max_msg_chars:
            first_line = first_line[:self._max_msg_chars - 1] + '…'
        self._records.append(ErrorRecord(op=str(op or '?'), msg=first_line))
        self._total_recorded += 1

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._records)

    @property
    def total_recorded(self) -> int:
        """Lifetime count, including records that have already aged out."""
        return self._total_recorded

    def recent(self) -> list[ErrorRecord]:
        return list(self._records)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def render_for_prompt(self) -> str:
        """Format the current records as a prompt block, or ``''`` if empty.

        Output shape (kept short and unambiguous):

            ## Recent failures to avoid (do not repeat these mistakes):
            1. [op=e2] ValueError: axis 1 is out of bounds for array of dim 1
            2. [op=m1] TypeError: unsupported operand type for /: ...
        """
        if not self._records:
            return ''
        lines = [self.HEADER]
        for i, rec in enumerate(self._records, start=1):
            lines.append(f'{i}. [op={rec.op}] {rec.msg}')
        return '\n'.join(lines)


# Convenience for tests / external rendering of an arbitrary record list.
def render_records(
        records: Iterable[ErrorRecord],
        header: str = ErrorMemory.HEADER,
) -> str:
    records = list(records)
    if not records:
        return ''
    lines = [header]
    for i, rec in enumerate(records, start=1):
        lines.append(f'{i}. [op={rec.op}] {rec.msg}')
    return '\n'.join(lines)
