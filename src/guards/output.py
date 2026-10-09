"""输出侧护栏：敏感信息脱敏 + 引用溯源校验。

两件事解决两类风险：
  - 脱敏：防止模型在回答里复述检索到的 PII（手机号、身份证、银行卡等）
  - 溯源：检查回答是否真的被检索片段支撑。这一层是"防御幻觉"，
    同时也间接防住了"给模型喂一篇伪造文档、让它据此回答"的投毒路径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.core.types import Action, ScoredChunk, Verdict

_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("手机号", re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), r"1**********"),
    ("身份证", re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "[身份证已脱敏]"),
    ("邮箱", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[邮箱已脱敏]"),
    ("银行卡", re.compile(r"(?<!\d)\d{16,19}(?!\d)"), "[卡号已脱敏]"),
    ("IPv4", re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), "[IP已脱敏]"),
]


@dataclass(frozen=True)
class RedactionResult:
    text: str
    hits: tuple[str, ...]


class PIIDetector:
    def redact(self, text: str) -> RedactionResult:
        hits: list[str] = []
        for name, pat, repl in _PATTERNS:
            if pat.search(text):
                hits.append(name)
                text = pat.sub(repl, text)
        return RedactionResult(text=text, hits=tuple(hits))


class OutputGuard:
    """把脱敏与溯源校验串起来，输出统一 Verdict。"""

    name = "output"

    def __init__(self, *, min_grounding: float = 0.25, strict: bool = False) -> None:
        self.pii = PIIDetector()
        self.min_grounding = min_grounding
        self.strict = strict  # True 时低支撑度直接拒绝，False 时降级为追加免责说明

    # ------------------------------------------------------------------
    @staticmethod
    def grounding_score(answer: str, contexts: list[ScoredChunk]) -> float:
        """用「回答中的实词有多少出现在任一被引用片段里」近似支撑度。

        这是廉价近似，不是严格的事实一致性判定。真机上可以换成 NLI 模型，
        接口的输入输出不变。
        """
        if not answer or not contexts:
            return 0.0
        from src.retriever.base import tokenize

        answer_terms = {t for t in tokenize(answer) if len(t) >= 2}
        if not answer_terms:
            return 0.0
        doc_terms: set[str] = set()
        for c in contexts:
            doc_terms |= set(tokenize(c.chunk.text))
        return round(len(answer_terms & doc_terms) / len(answer_terms), 3)

    # ------------------------------------------------------------------
    def check(self, answer: str, contexts: list[ScoredChunk]) -> Verdict:
        red = self.pii.redact(answer)
        score = self.grounding_score(red.text, contexts)

        notes: list[str] = []
        if red.hits:
            notes.append("已脱敏：" + "、".join(red.hits))

        if score < self.min_grounding:
            notes.append(f"引用支撑度偏低（{score:.2f} < {self.min_grounding}）")
            if self.strict:
                return Verdict(
                    action=Action.BLOCK, reason="回答缺乏引用支撑，已拒绝输出",
                    risk=1.0 - score, detail={"grounding": score, "notes": notes},
                )
            return Verdict(
                action=Action.SANITIZE,
                reason="；".join(notes) or "低支撑度",
                risk=1.0 - score,
                payload=red.text + "\n\n（提示：该回答在检索资料中的支撑度较低，请核验。）",
                detail={"grounding": score, "notes": notes},
            )

        return Verdict(
            action=Action.ALLOW,
            reason="；".join(notes) or "输出校验通过",
            risk=round(1.0 - score, 3),
            payload=red.text,
            detail={"grounding": score, "notes": notes},
        )
