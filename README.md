# 🚀 **Milvus Phrase Match Demos（适配 Milvus 2.6）**

高级短语匹配 × 多语言分析 × 日志检索 × RAG 检索增强

> 这个仓库展示了 Milvus 2.6 全新的 Phrase Match 能力：
> **支持短语匹配、词序匹配、距离(slop)容忍、多语言分词、中英文混合场景**
> 适用于搜索系统、企业知识库、日志检索、RAG 等真实工程场景。

---

# 📌 **仓库概览**

```
milvus-phrase-match-demos/
│
├── phrase_match_logs_demo.py        # 日志短语匹配：展示 slop=0~3 的实际效果
├── phrase_match_multilang_demo.py   # 中英混合短语匹配：展示多语言 + 分词器 + slop
├── requirements.txt                 # 依赖
└── README.md                        # 当前文件
```

---

# 🌟 **Phrase Match 是什么？为什么你需要它？**

Milvus 2.6 引入的 Phrase Match 是一项搜索核心能力：

> **基于倒排索引 + 分词 + 位置(position) + slop 的短语匹配技术。**

它能解决你在真实项目中 80% 的检索问题：

* 错误日志中短语必须出现（词序不能乱）
* 中文/英文短语是否连续决定了结果准确性
* “机器 学习 模型”和“机器 深度 学习 模型”都应视为学术术语变体
* 向量搜索无法表达“必须包含某短语”的硬约束
* RAG 系统需要：**短语必须在文档中出现（硬过滤） + 语义相关（向量排序）**

一句话总结：

> **Phrase Match = 你可以完全控制词序、间距、倒序、插词数量，非常适合真实工程场景。**

---

# 🔥 **Demo 1：错误日志短语匹配**（`phrase_match_logs_demo.py`）

日志场景是 Phrase Match 的“降维打击”应用。

假设这些日志：

```
connection reset by peer
connection fast reset by peer
connection was suddenly reset by the peer
peer reset connection by ...
peer unexpected connection reset happened
```

我们用 slop 展示短语匹配的容忍度：

* **slop=0：必须连续**
* **slop=1：允许插一个词**
* **slop=2：允许插多个词**
* **slop=3：允许倒序（中文英语通用）**

📌 运行`phrase_match_logs_demo.py`, 输出示例

<div align="center">
  <img src="phrase_match_logs_demo.png" alt="Phrase Match 搜索日志演示" width="800"/>
  <br>
  <em>图1: Phrase Match 搜索日志演示</em>
</div>

👉 这正是 BM25 / 向量检索都做不到的维度。

---

# 🌍 **Demo 2：多语言短语匹配（中文 × 英文）**（`phrase_match_multilang_demo.py`）

此 Demo 展示：

* 中文 Jieba analyzer 与英文 english analyzer 的行为差异
* slop=0/1/2/3 的匹配扩张过程
* 中文倒序句式 + 英文倒序句式
* 多语言 Phrase Match 的最佳工程实践

运行 `phrase_match_multilang_demo.py` 看看实际效果:

<div align="center">
  <img src="en.png" alt="英文示例" width="400"/>
  <img src="zh.png" alt="中文示例" width="400"/>
</div>

再重复一次：

👉 **英文与中文行为一致**
👉 **关键是你是否选了正确的 analyzer（english / chinese）**

---

# 🏗 **环境准备**

Python 环境：

```
pip install -r requirements.txt
```

requirements.txt：

```
pymilvus>=2.6
numpy
```

Milvus：

* 推荐 **Milvus 2.6 + Milvus Lite / docker / standalone**
* URI 默认：`http://localhost:19530`
* Token 默认：`root:Milvus`

---

# ▶️ 运行 Demo

### 运行日志示例：

```
python phrase_match_logs_demo.py
```

### 运行中英混合示例：

```
python phrase_match_multilang_demo.py
```

---

# 🧠 **Phrase Match 的 slop 应该如何选？（工程实践）**

这个仓库里包含了成熟的 slop 推荐策略：

| slop   | 适用场景             | 特点         |
| ------ | ---------------- | ---------- |
| **0**  | 错误日志排查 、API 名称、法律条款 | 最严格，需要连续短语 |
| **1**  | 技术手册、产品信息     | 允许插 1 个词   |
| **2**  | 中英文混合内容、RAG      | 常用匹配区间     |
| **3**  | 中文倒序较多的内容库、需要更宽松匹配    | 允许倒序       |
| **≥4** | 不推荐              | 成本高、意义小    |

总结：

> **slop 越小越严格，越大越宽松，超过 3 没必要。**

Milvus 2.6 之后，
**向量 + Phrase Match** 才是企业级搜索/RAG 的最佳实践组合。

欢迎直接跑 Demo，也欢迎 PR、提 issue，一起完善更多示例！



