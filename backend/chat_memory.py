from __future__ import annotations

from typing import List, Dict


class ChatMemory:
    def __init__(self):
        self._history: List[Dict[str, str]] = []

    def add_turn(self, question: str, answer: str) -> None:
        self._history.append({"question": question, "answer": answer})

    def as_context(self, limit: int = 5) -> str:
        recent = self._history[-limit:]
        if not recent:
            return ""
        context_lines: List[str] = []
        for turn in recent:
            context_lines.append(f"User: {turn['question']}")
            context_lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(context_lines)

    def clear(self) -> None:
        self._history.clear()
