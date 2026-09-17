---
AIGC:
    Label: "1"
    ContentProducer: 001191110102MACQD9K64018705
    ProduceID: 322602165630460_0-drive/220546414367104461/career_ai_transition/README_ai_rag_finance.md
    ReservedCode1: ""
    ContentPropagator: 001191110102MACQD9K64028705
    PropagateID: 322602165630460#1789659408709
    ReservedCode2: ""
---
# 金融研报 RAG 问答系统

> 基于真实研报 PDF 的 RAG（Retrieval-Augmented Generation，检索增强生成）问答系统
> 支持多 PDF 上传、语义检索、多轮对话、Web 可视化界面

一个从 0 到 1 完整落地的 RAG 应用：上传金融研报 PDF，即可用自然语言提问，系统自动检索相关内容并基于原文回答，不编造、可溯源。

---

## ✨ 功能特性

- 📄 **PDF 解析与管理**：支持上传多份研报 PDF，自动解析、切块、向量化
- 🔍 **语义检索**：基于 Embedding + FAISS 向量数据库，Top-K 相似度检索
- 🤖 **RAG 问答**：RetrievalQA Chain 封装，严格基于检索资料生成回答
- 💬 **多轮对话**：Session 会话管理，记住上下文，支持连续追问
- 🌐 **Web 界面**：FastAPI 后端 + 原生 HTML 前端，拖拽上传、对话气泡、参考资料折叠展示
- 🔄 **动态知识库**：上传/删除 PDF 后自动重建向量库，无需重启服务
- 📊 **引用溯源**：每条回答附带检索到的原文片段，可核对答案来源

---

## 🛠 技术栈

| 层级       | 技术                                    |
| ---------- | --------------------------------------- |
| 后端框架   | Python 3.9、FastAPI、Uvicorn            |
| RAG 框架   | LangChain、RetrievalQA Chain、LCEL      |
| 向量数据库 | FAISS（faiss-cpu）                      |
| Embedding  | 阿里云 DashScope `text-embedding-v3`    |
| LLM        | 通义千问 `qwen-plus`（OpenAI 兼容接口） |
| PDF 解析   | PyPDF                                   |
| 前端       | 原生 HTML + CSS + JavaScript（无框架）  |

---

## 🏗 系统架构

```
┌─────────────────────────────────────────────────┐
│                   前端 (HTML/JS)                  │
│   拖拽上传 PDF  │  对话界面  │  参考资料展示        │
└───────────────────────┬─────────────────────────┘
                        │ HTTP
┌───────────────────────▼─────────────────────────┐
│              FastAPI 后端服务 (:8002)             │
│                                                   │
│  /upload   /documents   /ask   /health            │
│                                                   │
│         ┌──────────────────────────┐              │
│         │   RetrievalQA Chain      │              │
│         │  (LangChain 封装)         │              │
│         └───────────┬──────────────┘              │
│                     │                             │
│        ┌────────────┴────────────┐                │
│        ▼                         ▼                │
│   FAISS 向量检索            通义千问 LLM            │
│   (Top-K 相似度)           (qwen-plus)            │
│        │                                          │
│   PDF → 切块 → Embedding(text-embedding-v3)       │
└───────────────────────────────────────────────────┘
```

**RAG 核心流程：**

```
用户提问
   ↓  Embedding 向量化
FAISS 相似度检索 Top-K
   ↓  检索到相关研报片段
拼入 Prompt（带对话历史）
   ↓
通义千问 LLM 生成回答
   ↓
回答 + 参考资料返回前端
```

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/lengchen141/ai-rag-finance.git
cd ai-rag-finance
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install langchain langchain-openai langchain-community \
            faiss-cpu dashscope fastapi uvicorn \
            python-multipart pypdf
```

### 3. 配置 API Key

编辑 `rag_api_server.py`，填入你的阿里云 DashScope API Key：

```python
os.environ["DASHSCOPE_API_KEY"] = "你的 DashScope Key"
os.environ["OPENAI_API_KEY"]    = "你的 DashScope Key"  # 同一个 Key
```

> DashScope Key 在[阿里云百炼控制台](https://bailian.console.aliyun.com/)获取。
> 两个环境变量填同一个 Key：前者供原生 Embedding SDK 使用，后者供 OpenAI 兼容的 LLM 接口使用。

### 4. 启动服务

```bash
python rag_api_server.py
```

看到以下输出说明启动成功：

```
📚 找到 N 个PDF文件
✂️  切块中...
🔢 向量化中（DashScope embedding）...
🤖 初始化RetrievalQA Chain...
✅ RAG服务初始化完成
INFO: Uvicorn running on http://0.0.0.0:8002
```

### 5. 使用系统

- **Web 界面**：浏览器打开 http://localhost:8002/
- **API 文档**（Swagger）：http://localhost:8002/docs

在网页上拖拽上传研报 PDF，即可开始提问。

---

## 📡 API 接口

| 方法   | 路径                               | 说明                       |
| ------ | ---------------------------------- | -------------------------- |
| GET    | `/ask?question=xxx&session_id=xxx` | 提问（支持多轮对话）       |
| POST   | `/upload`                          | 上传 PDF（自动重建知识库） |
| GET    | `/documents`                       | 列出已加载的 PDF           |
| DELETE | `/documents/{filename}`            | 删除指定 PDF               |
| GET    | `/health`                          | 健康检查                   |
| GET    | `/`                                | Web 前端页面               |

### 提问示例

```bash
curl "http://localhost:8002/ask?question=公司AI业务布局和进展？"
```

响应：

```json
{
  "question": "公司AI业务布局和进展？",
  "answer": "根据研报资料，公司在AI领域……",
  "session_id": "f3a2c1...",
  "sources": ["参考片段1...", "参考片段2...", "参考片段3..."],
  "history_count": 1
}
```

---

## 📁 项目结构

```
ai-rag-finance/
├── rag_api_server.py            # FastAPI 后端服务（RAG + 上传 + 会话）
├── rag_frontend.html            # Web 前端页面
├── langchain_embedding_rag.py   # 学习脚本：Embedding + FAISS 最小闭环
├── langchain_pdf_rag.py         # 学习脚本：真实 PDF 的 RAG（手搓版）
├── langchain_retrievalqa.py     # 学习脚本：RetrievalQA Chain 封装版
├── uploads/                     # 上传的 PDF 存放目录（gitignore）
├── .gitignore
└── README.md
```

---

## 🔑 关键技术点

- **文本切块策略**：`RecursiveCharacterTextSplitter`，按段落 → 行 → 句号 → 逗号递归切分，`chunk_size=300`、`overlap=50`，兼顾语义完整性与检索精度
- **Embedding 选型**：通义 `text-embedding-v3`（1024 维），中文金融语料效果好
- **检索模式**：FAISS `IndexFlatL2`，Top-K = 3
- **Chain 类型**：`RetrievalQA` + `chain_type="stuff"`，将检索片段一次性塞入 Prompt
- **防幻觉**：System Prompt 强制"仅基于参考资料回答，无相关信息则说明未提及"
- **多轮对话**：服务端按 `session_id` 维护对话历史，最近 6 条拼入 Prompt；前端用 localStorage 持久化会话

---

## 📈 学习路径

本项目是「Java 后端转 AI 应用工程师」学习计划中 W3（LangChain / RAG 阶段）的产出：

- **W1**：Python 基础、OOP、FastAPI、async/await、LLM API
- **W2**：LangChain 入门、PromptTemplate、LCEL、Tool 工具调用、手写 ReAct Agent
- **W3**：Embedding、FAISS、RAG 全流程、RetrievalQA、FastAPI 产品化（本项目）
- **W4-W5**：项目 ② —— Java Spring AI 版 RAG / 合规审查 Agent
- **W6+**：端到端全栈 AI 应用

作为 Java 后端工程师，本项目重点验证了：RAG 的工程本质（检索 + 拼接 + 生成）与语言无关，掌握 Python 实现后可平滑迁移到 Java Spring AI 技术栈。

---

## 🗺 后续规划

- [ ] 对话历史持久化（SQLite / Redis，替代内存存储）
- [ ] LLM 流式输出（SSE / WebSocket，打字机效果）
- [ ] 引用标注（答案定位到具体 PDF 页码）
- [ ] 混合检索（BM25 关键词 + 向量语义）+ Rerank 重排序
- [ ] 扫描件 PDF 支持（OCR）
- [ ] Docker 容器化部署

---

## 📄 License

MIT

---
