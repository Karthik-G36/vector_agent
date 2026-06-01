"""
OpenAI token usage tracker — module-level singleton.

Call record() after each OpenAI API response.
Call summary() at the end of the pipeline run to print a usage table.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class _Entry:
    label: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class _Tracker:
    _entries: List[_Entry] = field(default_factory=list)

    def record(
        self,
        label: str,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
    ) -> None:
        self._entries.append(
            _Entry(label, model, prompt_tokens, completion_tokens, total_tokens)
        )

    def summary(self) -> None:
        if not self._entries:
            return

        print("\n=== OpenAI Token Usage ===")
        col_label = max(len(e.label) for e in self._entries)
        col_model = max(len(e.model) for e in self._entries)

        header = (
            f"{'Call':<{col_label}}  {'Model':<{col_model}}"
            f"  {'Prompt':>8}  {'Completion':>10}  {'Total':>8}"
        )
        print(header)
        print("─" * len(header))

        for e in self._entries:
            print(
                f"{e.label:<{col_label}}  {e.model:<{col_model}}"
                f"  {e.prompt_tokens:>8,}  {e.completion_tokens:>10,}  {e.total_tokens:>8,}"
            )

        total = sum(e.total_tokens for e in self._entries)
        print("─" * len(header))
        print(f"{'Session total':<{col_label + col_model + 3}}  {'':>8}  {'':>10}  {total:>8,}")


    def count(self) -> int:
        return len(self._entries)

    def entries_from(self, index: int) -> List[_Entry]:
        """Return entries added since a snapshot index — used for per-job tracking."""
        return self._entries[index:]

    def as_dict(self, index: int = 0) -> dict:
        """Serialize entries since index into a JSON-friendly dict."""
        entries = self.entries_from(index)
        return {
            "total_tokens": sum(e.total_tokens for e in entries),
            "breakdown": [
                {
                    "label": e.label,
                    "model": e.model,
                    "prompt_tokens": e.prompt_tokens,
                    "completion_tokens": e.completion_tokens,
                    "total_tokens": e.total_tokens,
                }
                for e in entries
            ],
        }


# Module-level singleton — import and use directly.
tracker = _Tracker()
