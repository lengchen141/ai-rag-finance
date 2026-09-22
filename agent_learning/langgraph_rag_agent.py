"""
Agent + RAG 实战：让 Agent 自己决定要不要查研报知识库

架构：
    用户问题
       ↓
    Agent（LLM 决策）
       ├── 研报类问题 → search_research_reports 工具（RAG检索）
       ├── 计算类问题 → calculator 工具
       └── 闲聊/常识 → 不用工具，直接回答

Java 类比：
    两个 @Service（研报检索Service、计算Service）
    Controller 不写 if-else，让 LLM 当 Controller，看请求自己选 Service

安装：
    pip install langgraph langchain-openai langchain-community faiss-cpu pypdf

.env 文件里需要：
    DASHSCOPE_API_KEY=sk-xxx
    OPENAI_API_KEY=sk-xxx
"""

import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

# 固定动作：加载 .env（API Key）
load_dotenv()

PDF_PATH = "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"  # 你的研报文件名，没有就用模拟数据


# ============================================================
# 第一步：建研报知识库（和你 9/14 的 RAG 代码一样）
# ============================================================
def build_knowledge_base():
    """加载PDF建FAISS库；PDF不存在时用模拟金融文本兜底，保证脚本一定能跑"""
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
    )

    if os.path.exists(PDF_PATH):
        print(f"📄 加载真实PDF：{PDF_PATH}")
        loader = PyPDFLoader(PDF_PATH)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(pages)
    else:
        # 兜底：模拟研报文本（@formulas 里的 Document）
        print(f"⚠️ 没找到 {PDF_PATH}，使用模拟金融文本")
        from langchain_core.documents import Document
        mock_texts = [
            "恒生电子2025年净利润12.3亿元，同比增长18.5%。AI大模型业务收入占比从5%提升至15%，成为第二增长曲线。",
            "恒生电子发布LightGPT金融大模型，在研报解读、合规审查、智能客服三大场景实现商业化落地。",
            "平安银行上线140个运营审核类智能体，贷款审批时效从3天缩短至4小时，风险识别准确率提升34%。",
            "同花顺2025年营收89亿元，其中AI增值服务贡献占比达42%，iFinD智能终端用户付费率提升11个百分点。",
        ]
        chunks = [Document(page_content=t, metadata={"source": "模拟研报"}) for t in mock_texts]

    print(f"✂️ 共 {len(chunks)} 个文本块，向量化中...")
    db = FAISS.from_documents(chunks, embeddings)
    return db


# 全局知识库（启动时建一次，类似 Spring 的单例 Bean）
print("=" * 60)
print("启动中：构建研报知识库...")
vector_db = build_knowledge_base()
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
print("知识库就绪\n")


# ============================================================
# 第二步：定义工具（RAG检索 也是一个 Tool）
# ============================================================
@tool
def search_research_reports(query: str) -> str:
    """搜索金融研报知识库。当用户询问上市公司业绩、AI业务、金融行业数据、
    研报观点等需要具体资料的问题时使用。输入：查询问题。
    """
    # 这一步就是你熟悉的 RAG 检索
    docs = retriever.invoke(query)
    if not docs:
        return "知识库中没有找到相关内容。"
    # 把检索到的 chunks 拼成文本返回给 Agent
    results = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "未知来源")
        results.append(f"[资料{i}]（来源：{source}）\n{doc.page_content}")
    return "\n\n".join(results)


@tool
def calculator(expression: str) -> str:
    """数学计算器。当用户需要计算数值（加减乘除、百分比等）时使用。
    输入：数学表达式，如 "1000 * 52.3" 或 "(72-24) / 72 * 100"
    """
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


# ============================================================
# 第三步：创建 Agent（RAG检索 + 计算器 两个工具）
# ============================================================
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

tools = [search_research_reports, calculator]
agent = create_react_agent(llm, tools)


# ============================================================
# 第四步：观察 Agent 怎么决策
# ============================================================
def ask(question: str):
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print("=" * 60)

    result = agent.invoke({"messages": [("user", question)]})

    print("\n📋 Agent 决策过程:")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", [])
        if role == "AIMessage" and tool_calls:
            for tc in tool_calls:
                print(f"  → 决定调用工具：{tc['name']}，参数：{tc['args']}")
        elif role == "ToolMessage":
            print(f"  ← 工具返回：{msg.content[:120]}...")

    print(f"\n💬 最终回答：\n{result['messages'][-1].content}")


if __name__ == "__main__":
    # 场景1：研报类问题 → 应该调 search_research_reports
    ask("恒生电子的AI业务表现怎么样？")

    # 场景2：纯计算 → 应该只调 calculator，不查研报（省token、更快）
    ask("帮我算一下 1234 * 56 等于多少")

    # 场景3：闲聊 → 不调任何工具，直接回答
    ask("你好，你是谁？")

    # 场景4：组合问题 → 先查研报拿数据，再调计算器算百分比
    ask("平安银行的贷款审批时效，从3天缩短到4小时，请问缩短了百分之多少？")
