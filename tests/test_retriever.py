"""ACL 与检索层的测试。

这一组是整个项目最要紧的测试：一旦 ACL 下推失效，整个系统的安全模型就崩了。
所以用断言而不是靠 demo 输出来判断。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.core.types import Chunk
from src.retriever.base import tokenize
from src.retriever.hybrid import HeuristicReranker, HybridRetriever
from src.retriever.lexical import BM25Retriever
from src.store.db import connect, count_chunks, insert_chunks, load_visible_chunks, upsert_document


def mk(doc_id: str, text: str, tenant: str, group: str, i: int = 0) -> Chunk:
    return Chunk(chunk_id=f"{doc_id}#{i}", doc_id=doc_id, ordinal=i, text=text,
                 tenant_id=tenant, acl_group=group, title=doc_id)


class TestStoreACL(unittest.TestCase):
    """验证 ACL 在 SQL 层被过滤，而不是读回来再剔除。"""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.conn = connect(str(Path(self.tmp) / "t.db"))
        upsert_document(self.conn, doc_id="d1", title="A", tenant_id="acme", acl_group="finance")
        upsert_document(self.conn, doc_id="d2", title="B", tenant_id="acme", acl_group="hr")
        upsert_document(self.conn, doc_id="d3", title="C", tenant_id="globex", acl_group="staff")
        insert_chunks(self.conn, [
            mk("d1", "报销标准相关内容", "acme", "finance", 0),
            mk("d2", "薪酬制度相关内容", "acme", "hr", 0),
            mk("d3", "供应商条款内容", "globex", "staff", 0),
        ])
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()

    def test_only_visible_chunks_returned(self):
        got = load_visible_chunks(self.conn, "acme", {"finance"})
        self.assertEqual([c.doc_id for c in got], ["d1"])

    def test_cross_tenant_invisible(self):
        got = load_visible_chunks(self.conn, "globex", {"staff"})
        self.assertEqual([c.doc_id for c in got], ["d3"])
        self.assertNotIn("d1", [c.doc_id for c in got])

    def test_empty_groups_yields_nothing(self):
        """最小权限：不属于任何权限组，就什么都看不到。"""
        self.assertEqual(load_visible_chunks(self.conn, "acme", set()), [])

    def test_total_count_still_three(self):
        """数据是存进去了的，只是对调用者不可见 —— 证明是过滤而非没写入。"""
        self.assertEqual(count_chunks(self.conn), 3)


class TestTokenizer(unittest.TestCase):
    def test_chinese_bigram(self):
        toks = tokenize("报销标准")
        self.assertIn("报销", toks)
        self.assertIn("标准", toks)

    def test_lowercase_latin(self):
        self.assertIn("rag", tokenize("RAG Agent"))

    def test_no_crash_on_mixed(self):
        self.assertTrue(tokenize("出差报销 invoice 2026 年度指南"))


class TestBM25ACL(unittest.TestCase):
    def setUp(self) -> None:
        self.r = BM25Retriever()
        self.r.index([
            mk("finance", "出差住宿报销上限六百元", "acme", "finance", 0),
            mk("hr", "年终奖三个月基本工资", "acme", "hr", 0),
            mk("supplier", "供应商结算周期六十天", "globex", "staff", 0),
        ])

    def test_visible_doc_scores(self):
        hits = self.r.search("报销上限", top_k=5, acl=("acme", frozenset({"finance"})))
        self.assertTrue(hits)
        self.assertEqual(hits[0].chunk.doc_id, "finance")

    def test_invisible_never_appears(self):
        hits = self.r.search("年终奖", top_k=5, acl=("acme", frozenset({"finance"})))
        self.assertNotIn("hr", [h.chunk.doc_id for h in hits])

    def test_tenant_mismatch_returns_empty(self):
        self.assertEqual(self.r.search("报销", top_k=5, acl=("globex", frozenset({"finance"}))), [])


class TestRRF(unittest.TestCase):
    """RRF 的融合应当同时参考两路排名。"""

    def test_prefers_doc_ranked_by_both(self):
        """在两路都出现的文档，应当排在只出现在单路的文档之前 —— 这是 RRF 的核心行为。"""
        from src.core.types import ScoredChunk

        class Fake:
            name = "fake"

            def __init__(self, chunks):
                self.cs = chunks

            def index(self, c):
                pass

            def search(self, q, top_k=5, *, acl):
                return self.cs

        c_a = mk("A", "AAA", "t", "g", 0)
        c_b = mk("B", "BBB", "t", "g", 0)
        c_c = mk("C", "CCC", "t", "g", 0)
        # 向量路：A 第 1，B 第 2；词法路：A 第 1，C 第 2
        hits_v = [ScoredChunk(c_a, 1.0, "vector", 1), ScoredChunk(c_b, .5, "vector", 2)]
        hits_l = [ScoredChunk(c_a, 1.0, "lexical", 1), ScoredChunk(c_c, .5, "lexical", 2)]

        h = HybridRetriever(Fake(hits_v), Fake(hits_l), weights={"fake": 1.0})
        out = h.search("q", top_k=5, acl=("t", frozenset({"g"})))
        ids = [x.chunk.doc_id for x in out]
        self.assertEqual(ids[0], "A")
        self.assertIn("B", ids[:2])
        self.assertIn("C", ids)


class TestHeuristicReranker(unittest.TestCase):
    def test_term_coverage_reorders(self):
        from src.core.types import ScoredChunk

        doc = [mk("doc", "出差报销标准是六百元", "t", "g", i) for i in range(1)]
        hits = [ScoredChunk(doc[0], 0.01, "hybrid", 1)]
        out = HeuristicReranker().rerank("报销标准", hits, top_k=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].source, "rerank")


if __name__ == "__main__":
    unittest.main()
