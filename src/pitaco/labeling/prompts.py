"""Prompt assembly. Prompts live as versioned files in `prompts/`; nothing is hard-coded here.

The teacher system prompt is one stable prefix (instructions + examples) followed by the review in
the user turn. A stable prefix is what lets providers cache it, so the long few-shot part is paid
in full only once.
"""

import json
from pathlib import Path
from typing import Literal

Mode = Literal["zeroshot", "fewshot"]


def load_fewshot(prompts_dir: Path, version: str) -> list[dict]:
    path = prompts_dir / f"fewshot_{version}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_example(text: str, output: dict) -> str:
    return f"Avaliação: {text}\nJSON: {json.dumps(output, ensure_ascii=False)}"


def teacher_system_prompt(prompts_dir: Path, version: str, mode: Mode) -> str:
    instructions = (prompts_dir / f"teacher_{version}.md").read_text(encoding="utf-8").strip()
    if mode == "zeroshot":
        return instructions
    examples = "\n\n".join(
        render_example(ex["text"], ex["output"]) for ex in load_fewshot(prompts_dir, version)
    )
    return f"{instructions}\n\nExemplos:\n\n{examples}"


def student_instruction(prompts_dir: Path, version: str) -> str:
    return (prompts_dir / f"student_{version}.md").read_text(encoding="utf-8").strip()


def chat_messages(system_prompt: str, review_text: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Avaliação: {review_text}"},
    ]
