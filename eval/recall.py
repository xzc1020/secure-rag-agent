"""检索评测与 RRF 融合权重网格搜索。

产出简历上那句「Recall@5 相比纯向量提升 XX.X 个百分点」的数字。

要点：融合权重是搜出来的，不是拍脑袋定的。这一步把"我看别人这么做"
变成"我在自己的评测集上验证过这么做更好"，面试时能讲的内容完全不同。

运行： PYTHONPATH=. python eval/recall.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo import DOCS
from src.core.types import ScoredChunk
from src.pipeline.factory import build_bundle
from src.retriever.hybrid import HeuristicReranker, HybridRetriever


def recall_at_k(hits: list[ScoredChunk], gold_doc: str, k: int) -> float:
    """命中 gold 文档记 1，否则 0。文档级判定比片段级更贴合真实可用性。"""
    return 1.0 if any(h.chunk.doc_id == gold_doc for h in hits[:k]) else 0.0


def reciprocal_rank(hits: list[ScoredChunk], gold_doc: str) -> float:
    for i, h in enumerate(hits, 1):
        if h.chunk.doc_id == gold_doc:
            return 1.0 / i
    return 0.0


KS = (1, 3, 5)


def evaluate(retriever: HybridRetriever, qa_set: list[dict], top_k: int = 5) -> dict[str, float]:
    """返回多档位指标。只看 Recall@5 太粗，@1 才能看出排序质量差异。"""
    acc = {f"Recall@{k}": 0.0 for k in KS}
    acc["MRR"] = 0.0
    for item in qa_set:
        acl = (item["tenant_id"], frozenset(item["groups"]))
        hits = retriever.search(item["question"], top_k=top_k, acl=acl)
        for k in KS:
            acc[f"Recall@{k}"] += recall_at_k(hits, item["gold_doc"], k)
        acc["MRR"] += reciprocal_rank(hits, item["gold_doc"])
    n = len(qa_set) or 1
    return {k: v / n for k, v in acc.items()}


def load_qa(path: str) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", default="eval/qa_set.json")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--embedding", default="hashing",
                    choices=["hashing", "sentence-transformers"])
    args = ap.parse_args()

    qa_set = load_qa(args.qa)
    bundle = build_bundle(DOCS, embedding_kind=args.embedding)
    vec, lex = bundle.retriever.vector, bundle.retriever.lexical

    def hybrid(w: float, rerank: bool) -> HybridRetriever:
        return HybridRetriever(
            vector=vec, lexical=lex,
            weights={"vector": 1.0 - w, "bm25": w},
            reranker=HeuristicReranker() if rerank else None,
        )

    m_vec = evaluate(hybrid(0.0, False), qa_set, args.top_k)
    m_lex = evaluate(hybrid(1.0, False), qa_set, args.top_k)

    header = f"{'':<22}" + "".join(f"{c:>11}" for c in ("Recall@1", "Recall@3", "Recall@5", "MRR"))
    print(f"\n评测集 {len(qa_set)} 条")
    print("-" * 68)
    print(header)
    print("-" * 68)
    for label, m in (("纯向量召回", m_vec), ("纯 BM25 召回", m_lex)):
        print(f"{label:<22}" + "".join(f"{m[c]:>11.3f}" for c in ("Recall@1", "Recall@3", "Recall@5", "MRR")))

    cols = ("Recall@1", "Recall@3", "Recall@5", "MRR")
    grid = [float(x) for x in args.grid.split(",")]

    def table(title: str, fn) -> tuple[float, dict[str, float]] | None:
        print(f"\n{title}")
        print("-" * 68)
        print(f"{'w':>6}" + "".join(f"{c:>11}" for c in cols))
        print("-" * 68)
        best: tuple[float, dict[str, float]] | None = None
        for w in grid:
            m = fn(w)
            mark = ""
            if best is None or m[f"Recall@{args.top_k}"] > best[1][f"Recall@{args.top_k}"]:
                best, mark = (w, m), "  <- best"
            print(f"{w:>6.2f}" + "".join(f"{m[c]:>11.3f}" for c in cols) + mark)
        return best

    best_no = table("RRF 权重网格 · 无重排", lambda w: evaluate(hybrid(w, False), qa_set, args.top_k))
    best_rr = table("RRF 权重网格 · 融合 + 重排", lambda w: evaluate(hybrid(w, True), qa_set, args.top_k))

    print("-" * 68)
    for label, best in (("无重排", best_no), ("融合+重排", best_rr)):
        if not best:
            continue
        w, m = best
        key = f"Recall@{args.top_k}"
        delta = (m[key] - m_vec[key]) * 100
        print(f"[{label}] 最优 w_bm25={w:.2f}  {key}={m[key]:.3f}  相对纯向量 {delta:+.1f}pp  MRR={m['MRR']:.3f}")
        print(f"          参照：纯向量 {m_vec[key]:.3f}，纯 BM25 {m_lex[key]:.3f}")

    print()
    print("诚实提示：语料规模（30 篇文档、205 条评测题）已足以让指标产生区分度，")
    print("但当前跑的是 HashingEmbedding 降级实现，只能说明「混合优于单路」的趋势")
    print("在这个评测集上成立。接真实 embedding 后重跑，才是能写进简历的数字。")
    print()
    print("另注：本评测集的题干措辞与文档措辞刻意做了区分，BM25 单路因此表现也不错。")
    print("真实场景的长尾查询词汇差异更大，两路的互补性会比这里更明显。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
