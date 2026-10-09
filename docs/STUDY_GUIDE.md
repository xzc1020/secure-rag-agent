# 学习手册

这份文档服务于一个目标：**让你不靠背诵也能讲清楚这个项目**。

`ARCHITECTURE.md` 讲"为什么这么设计"，`INTERVIEW.md` 讲"面试怎么答"，
这份讲"**怎么把它变成你自己的**"。

---

## 一、学习路径（建议 5 天，每天 2-3 小时）

不要一上来就读代码。按这个顺序：

| 天 | 做什么 | 验收标准 |
|---|---|---|
| **第 1 天** | 跑通 `demo.py`，把四个场景的输出读完；再跑 `eval/recall.py` 和 `eval/attacks.py` | 能说出四个场景分别在验证什么 |
| **第 2 天** | 读 `src/core/types.py` 和 `src/store/db.py`，然后读本文第三节的第 1、2 条讲解 | 能画出一个查询从输入到输出的数据流 |
| **第 3 天** | 读 `src/retriever/` 全部四个文件，对照本文第三节第 3、4 条 | 能解释 RRF 解决了什么问题、代价是什么 |
| **第 4 天** | 读 `src/guards/` 两个文件，然后做完本文第四节的全部实验 | 能说出一个自己改出来的数字变化 |
| **第 5 天** | 读 `src/core/graph.py` 和 `src/pipeline/`，做本文第五节的自测题 | 20 题能答对 15 题以上 |

**第 4 天最关键。** 那天做完实验，这个项目才真正开始属于你。

---

## 二、逐文件导读

按重要性分三档。★★★ 必须完全读懂，★★ 要理解设计意图，★ 知道存在即可。

| 文件 | 档 | 一句话 | 你该掌握什么 |
|---|---|---|---|
| `src/store/db.py` | ★★★ | 存储层，ACL 下推发生在这里 | SQL WHERE 怎么写、索引为什么这么建 |
| `src/retriever/lexical.py` | ★★★ | BM25 词法召回 | 倒排表遍历时如何跳过不可见数据 |
| `src/retriever/hybrid.py` | ★★★ | RRF 融合 + 重排 | 融合公式、归一化为什么必须做 |
| `src/guards/input.py` | ★★★ | 输入护栏 | 归一化防御、三级处置、风险分怎么算 |
| `src/core/types.py` | ★★★ | 跨模块契约 | 每个字段为什么这么设计 |
| `src/retriever/vector.py` | ★★ | 向量召回 | 暴力余弦的适用边界、增量编码 |
| `src/core/graph.py` | ★★ | 编排引擎 | 选边优先级、审计日志怎么合并 |
| `src/guards/output.py` | ★★ | 输出护栏 | 脱敏规则、溯源是近似不是判定 |
| `src/pipeline/rag_graph.py` | ★★ | 主流程 | 节点契约、被拦截时如何短路 |
| `src/models.py` | ★★ | 模型适配层 | 三个实现各自的适用场合 |
| `src/retriever/base.py` | ★★ | 协议 + 分词 | 为什么用 bigram 不用分词器 |
| `src/pipeline/factory.py` | ★ | 装配 | 换实现改哪里 |
| `src/config.py` | ★ | 配置加载 | 为什么用 JSON 不用 YAML |
| `src/ingest/pipeline.py` | ★ | 灌库切片 | 为什么不固定长度硬切 |
| `app/api.py` / `app/ui.py` | ★ | 服务与界面 | 不是项目重点 |
| `eval/*.py` | ★★ | 评测脚本 | **数字的唯一来源，必须会跑会改** |
| `tools/expand_attacks.py` | ★★ | 样本变异 | 两个方法论陷阱 |

---

## 三、六个核心机制，逐段讲

### 1. ACL 下推（`store/db.py`）

```python
def load_visible_chunks(conn, tenant_id, groups):
    if not groups:
        return []                      # ① 最小权限
    placeholders = ",".join("?" for _ in groups)
    sql = ("SELECT ... FROM chunks "
           f"WHERE tenant_id = ? AND acl_group IN ({placeholders}) "
           "ORDER BY doc_id, ordinal")
    return [Chunk(**dict(r)) for r in conn.execute(sql, [tenant_id, *groups])]
```

三个要点：

- **① 空组直接返回空**。不属于任何权限组 = 什么都看不到。同时也避免了 `IN ()` 的 SQL 语法错误。
- **② 过滤在 SQL 层**。不可见的行根本不会进入 Python 内存。
- **③ 索引前缀匹配**。`idx_chunks_acl ON chunks(tenant_id, acl_group)` 正好覆盖这个 WHERE。

配套的 schema 设计要点：`tenant_id` 和 `acl_group` 是**冗余**落在 `chunks` 表上的。
不冗余的话，每次检索都要 join `documents` 表，向量检索场景下代价很高。
代价是改文档权限时要同步更新它的所有切片 —— 这是典型的读多写少场景下的空间换时间。

> **怎么验证它真的生效？**
> 测试 `test_total_count_still_three` 专门验证：库里确实有 3 条数据，
> 但对某个调用者只返回 1 条。是"过滤"而不是"没写入"。

---

### 2. BM25 里的 ACL（`retriever/lexical.py`）

```python
def search(self, query, top_k=20, *, acl):
    visible = self._visible_indices(acl)      # 缓存：ACL → 可见下标
    visible_set = set(visible)
    ...
    for i in postings:
        if i not in visible_set:              # ← 在这里下推
            continue
```

这里没有 SQL，所以下推表现为**遍历倒排表时跳过**。和向量侧的 SQL WHERE 是两种实现，
但语义等价：不可见的 chunk 既不参与打分，也不会进结果集。

`_visible_indices()` 做了缓存，避免每次请求都重新扫一遍全表判断可见性。
缓存 key 是 `(tenant_id, frozenset(groups))` —— 这也是为什么 `groups` 必须设计成
`frozenset` 而不是 `list`（要能当字典 key）。

---

### 3. RRF 融合（`retriever/hybrid.py`）

```python
for source, hits in ((self.vector.name, v_hits), (self.lexical.name, l_hits)):
    w = self.weights.get(source, 1.0)
    for h in hits:
        fused[h.chunk.chunk_id] += w / (self.rrf_k + h.rank)
```

**为什么这样算？** BM25 输出 8.3，余弦输出 0.87，两者量纲不可比。
要加权就得先归一化，而归一化系数对数据分布敏感。RRF 只用排名，免疫量纲。

**代价是什么？** 排名第 1 和第 2 的差距被压平了（1/61 vs 1/62 差得很小），
掉了绝对相关性信息。所以后面要接一层 rerank 补回来。

**`HeuristicReranker` 的归一化（踩过的坑）**：

```python
# 错误写法：融合分约 0.016，覆盖率 0~1，直接加权等于融合分没用上
final = 0.7 * overlap + 0.2 * h.score + 0.1 * position_bonus

# 正确写法：三项各自归一化后再加权
final = (0.6 * covers[i] / c_max
       + 0.3 * scores[i] / s_max
       + 0.1 * positions[i] / p_max)
```

最初的版本导致所有权重配置跑出完全相同的指标 —— 因为融合分那一项等于没参与计算。
**这就是一个现成的"我改过什么"的故事。**

---

### 4. 输入护栏（`guards/input.py`）

执行顺序：`normalize() → 规则匹配 → 算风险分 → 按阈值定处置`

**归一化放在最前面**（NFKC + 剥离零宽字符）：

```python
def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    return _ZERO_WIDTH.sub("", text)
```

NFKC 把全角转回半角；再手动剥离零宽字符。不做这一步，
"Ｉｇｎｏｒｅ" 和 "Ignore" 是两个不同字符串，规则被轻易绕过。

有意思的推论：**归一化本身就是一种防御**。所以扩充工具生成全角/零宽变体时，
在语义去重后它们会退化成同一条 —— 测试里专门有两条验证这一点。

**风险分与三级处置**：

| 风险分 | 处置 | 说明 |
|---|---|---|
| < 0.35 | 放行 | |
| 0.35 ~ 0.70 | 清洗后放行 | 剥离标签式载荷后继续服务 |
| ≥ 0.70 | 拒绝 | |

多类规则同时命中会加权（每多一类 +0.25），因为多类命中说明这是构造过的载荷。
混淆编码（长 base64 / hex / unicode 转义）是弱信号，单独出现只 +0.15。

---

### 5. 输出护栏（`guards/output.py`）

```python
def grounding_score(answer, contexts):
    answer_terms = {t for t in tokenize(answer) if len(t) >= 2}
    doc_terms = set()
    for c in contexts:
        doc_terms |= set(tokenize(c.chunk.text))
    return len(answer_terms & doc_terms) / len(answer_terms)
```

**这是廉价近似，不是严格的事实一致性判定。**
它测的是"回答里的词有多少出现在检索到的片段里"，会有假阳性：
回答换了说法但意思对，分数也会低。真正的做法是用 NLI 模型。

`strict` 参数控制低支撑度时是直接拒绝还是追加免责说明。默认 false，
因为直接拒绝会把"资料里确实有但表述不同"的正常回答误杀。

---

### 6. 编排引擎（`core/graph.py`）

```python
g.edge("guard_input", "retrieve")
g.edge("guard_input", "refuse", when=lambda c: c.blocked)
```

选边优先级：**显式 `when` → `label` → 默认边**。

被拦截时 `ctx.blocked = True`，`when` 命中，走 `refuse` 分支。
**检索节点不会执行** —— 这就是 demo 场景 ④ 里"检索到 0 条"的原因。

审计日志的合并逻辑（踩过坑）：初版是节点内部记一条、引擎再记一条。
现在改成引擎记录执行前的 `len(ctx.audit)`，只给新增的那几条补 `cost_ms` 和 `node`。
节点没记日志时引擎才补一条。

---

## 四、动手实验（第 4 天必做）

**每个实验都要记录数字的前后变化。** 这些变化就是你面试时的素材。

### 实验 1：加一条自己的规则

1. 先跑 `python eval/attacks.py`，记下拦截率，并把"未被处置的样本"看一遍
2. 挑 3-5 条你红队经验里见过但规则库没有的表述，加进 `_RULES`
3. 重新跑，看拦截率变化多少、误拒率有没有跟着涨

**这个实验会让你亲身体会权衡**：拦截率上去的同时误拒率往往也上去。
（我今天就踩了一次：把"忘记"后面的敏感名词放宽成可选，
误拒率从 0% 涨到 2.5%，把"我忘记了之前提交的报销单"误判成攻击。）

### 实验 2：调护栏阈值

```bash
python eval/attacks.py --sweep
```

看那张权衡表。**不要挑拦截率最高的那一行**，挑误拒率你能接受的前提下拦截率最高的。

### 实验 3：破坏检索

把 `config.json` 里 `retrieval.weights` 改成 `{"vector": 1.0, "bm25": 0.0}`，
再跑 `eval/recall.py`，观察 Recall@1 从 0.951 掉到 0.902。

然后把 `rerank` 改成 `false`，再看一遍。

### 实验 4：验证 ACL 真的有效

Gradio 界面里，同一个问题"公司的年终奖怎么算？"，
分别在 `finance` 组和 `hr` 组下问，看检索到的片段完全不同。

或者写代码验证：

```python
from src.config import load_config, docs_from_config
from src.pipeline.factory import build_from_config
from src.pipeline.rag_graph import answer

b = build_from_config(load_config())
q = "公司的年终奖怎么算？"
for groups in ({"public", "finance"}, {"public", "hr"}):
    ctx = answer(q, tenant_id="acme", groups=groups, graph=b.graph)
    print(groups, "→", {h.chunk.doc_id for h in ctx.candidates})
```

### 实验 5：扩充种子，观察"种子多样性"效应

```bash
python tools/expand_attacks.py --target 820 --indirect-ratio 0.4
```

先用现有的 130 条种子跑，看能生成多少；
再手动往 `eval/attack_set.json` 里加 20 条你自己写的种子，看生成量怎么变。

**你会直观感受到"种子多样性决定样本多样性"这句话不是空话。**

---

## 五、自测题（第 5 天）

不看代码，口头回答。答不上来的回去重读对应章节。

**基础（必须全对）**

1. 一次请求从输入到输出，经过哪几个节点？
2. ACL 过滤发生在哪一层？为什么不能在应用层做？
3. `groups` 为什么是 `frozenset` 而不是 `list`？
4. 被拦截的请求会产生几次检索？为什么？
5. 项目里有两个租户、几个权限组？语料有多少篇文档？

**进阶（答对 8 题以上）**

6. RRF 相比加权求和解决了什么问题？代价是什么？
7. 融合权重是怎么确定的？
8. `HeuristicReranker` 初版有什么 bug？怎么修的？
9. 为什么中文分词用 bigram 而不是分词器？
10. 为什么 ACL 字段要冗余到 chunks 表？
11. 护栏为什么要三级处置而不是一刀切拒绝？
12. 归一化为什么放在规则匹配之前？
13. 引用溯源是严格的事实判定吗？
14. 攻击样本集有哪些类别？各多少条？
15. 拦截率和误拒率分别怎么算？

**挑战（答对 3 题以上就够了）**

16. 数据量到十万级，你会怎么改？
17. 换 pgvector 需要动哪些地方？
18. 攻击样本变异时，为什么编码类要单独标记？
19. 为什么 30 条种子变不出 820 条有信息量的样本？
20. 现在这个项目最大的短板是什么？你打算怎么补？

---

## 六、常见困惑

**Q：代码里到处是"降级实现"，这是不是说明项目不完整？**

不是。这是刻意的分层：`HashingEmbedding` / `MockLLM` 让项目零依赖可跑，
真实模型通过适配层替换。**取舍本身就是设计的一部分**，面试时要主动讲出来。

**Q：为什么拦截率只有 50% 左右，看起来很差？**

因为规则层有结构性局限，改写的攻击认不出来。
这个数字是真实的，而且它正好说明了为什么要接模型通道。
**一个诚实的 52% 加清晰的归因，比虚高的 90% 有价值得多。**

**Q：README 里说数字"不能写进简历"，那我写什么？**

写结构、写设计、写方法论。等接了真实 embedding 重跑之后，再补数字。
`PLAN.md` 里有三阶段的简历写法。

**Q：我应该先补哪一块？**

按投入产出比排序：

1. **接真实 embedding**（改 config 一个字段 + 装 Ollama）→ 立刻得到可用的 Recall 数字
2. **补攻击种子**（你的红队经验直接变现）→ 拦截率会明显上升
3. **接模型通道护栏** → 攻击成功率显著下降，这是最大的技术增量
