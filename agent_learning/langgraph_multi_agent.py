"""
Multi-Agent Supervisor 模式：老板Agent分派任务给专家Agents

架构：
    用户问题
       ↓
    Supervisor（老板，只做路由决策）
       ├──→ ResearchAnalyst（研报分析师）：分析业务/公司/行业
       ├──→ DataAnalyst（数据分析师）：分析数字/财务指标
       ──→ Summarizer（总结专家）：整合结论、写报告

Java 类比：
    单Agent多工具 = 一个大Controller干所有事（反模式）
    Supervisor模式 = 职责清晰的分层架构
    Supervisor ≈ 路由Controller
    Worker ≈ 专业Service，各有独立prompt和能力边界

环境变量：
    OPENAI_API_KEY=sk-xxx
"""

import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()


# ============================================================
# 第一步：共享状态（所有Agent读写的"白板"）
# ============================================================
class AgentState(TypedDict):
    messages: List          # 对话历史
    next_worker: str        # Supervisor决策：下一步派谁
    final_answer: str       # 最终答案


# ============================================================
# 第二步：三个专家Agent（各有独立system prompt）
# ============================================================
def make_llm():
    return ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )

RESEARCH_PROMPT = """你是资深金融研报分析师。
能力：分析公司业务、竞争优势、行业地位、AI布局、风险。
输出：2-3条关键洞察，每条带依据。"""

DATA_PROMPT = """你是金融数据分析师，专注数字和指标。
能力：财务数据解读、同比环比计算、指标提取。"""

SUMMARIZE_PROMPT = """你是报告撰写专家。
任务：把前面各专家分析整合成简洁结论。
格式：要点式，5行以内，给决策者看。"""


def get_user_question(state: AgentState) -> str:
    """从messages提取最近的用户问题"""
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return ""


def get_analysis_context(state: AgentState) -> str:
    """汇总前面各专家的分析结果"""
    return "\n".join([
        m.content for m in state["messages"]
        if isinstance(m, AIMessage) and m.content.startswith("[")
    ])


def research_analyst_node(state: AgentState) -> AgentState:
    """研报分析专家"""
    print("\n [研报分析师] 分析中...")
    msg = make_llm().invoke([
        SystemMessage(content=RESEARCH_PROMPT),
        HumanMessage(content=get_user_question(state)),
    ])
    print(f"   {msg.content[:150]}...")
    return {
        "messages": state["messages"] + [AIMessage(content=f"[研报分析]\n{msg.content}")],
        "next_worker": "data",       # 固定流转：研报→数据
        "final_answer": "",
    }


def data_analyst_node(state: AgentState) -> AgentState:
    """数据分析专家"""
    print("\n📊 [数据分析师] 分析中...")
    context = get_analysis_context(state)
    question = get_user_question(state)
    msg = make_llm().invoke([
        SystemMessage(content=DATA_PROMPT),
        HumanMessage(content=f"问题：{question}\n\n前面分析：\n{context}"),
    ])
    print(f"   {msg.content[:150]}...")
    return {
        "messages": state["messages"] + [AIMessage(content=f"[数据分析]\n{msg.content}")],
        "next_worker": "summarize",  # 数据→总结
        "final_answer": "",
    }


def summarizer_node(state: AgentState) -> AgentState:
    """总结专家，输出最终答案"""
    print("\n📝 [总结专家] 整合报告...")
    analysis = get_analysis_context(state)
    msg = make_llm().invoke([
        SystemMessage(content=SUMMARIZE_PROMPT),
        HumanMessage(content=f"基于以下分析整合结论：\n\n{analysis}"),
    ])
    print(f"   {msg.content}")
    return {
        "messages": state["messages"] + [AIMessage(content=msg.content)],
        "next_worker": "end",
        "final_answer": msg.content,
    }


# ============================================================
# 第三步：Supervisor（老板，只做路由）
# ============================================================
SUPERVISOR_PROMPT = """你是任务调度器。根据用户问题，决定第一步派哪个专家。

可用专家：
- research：研报/业务/公司/行业/战略分析问题
- data：纯数字/计算/财务指标/增长率问题

重要规则：
1. 如果是"总结/综合/写报告"类问题，**必须先派 research**，
   后续会自动流转到 data 和 summarize。不要直接跳到最后。
2. 纯数字计算才派 data。
3. 只输出一个词：research 或 data
"""


def supervisor_node(state: AgentState) -> AgentState:
    """老板：看问题，决定第一步派谁"""
    question = get_user_question(state)
    print(f"\n [Supervisor] 收到问题：{question[:60]}")

    msg = make_llm().invoke([
        SystemMessage(content=SUPERVISOR_PROMPT),
        HumanMessage(content=question),
    ])

    decision = "data" if "data" in msg.content.strip().lower() else "research"
    print(f"   决策：第一步派给 {decision}")
    return {
        "messages": state["messages"],
        "next_worker": decision,
        "final_answer": "",
    }


# ============================================================
# 第四步：条件路由
# ============================================================
def route_after_supervisor(state: AgentState) -> str:
    return state["next_worker"]


# ============================================================
# 第五步：组装状态机
# ============================================================
def build_multi_agent():
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_analyst_node)
    graph.add_node("data", data_analyst_node)
    graph.add_node("summarize", summarizer_node)

    # 开始→老板
    graph.add_edge(START, "supervisor")

    # 老板决策后路由
    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"research": "research", "data": "data"},
    )

    # 固定流水线：研报→数据→总结
    graph.add_edge("research", "data")
    graph.add_edge("data", "summarize")

    # 总结→结束
    graph.add_edge("summarize", END)

    return graph.compile()


# ============================================================
# 第六步：测试
# ============================================================
def ask(question: str):
    print(f"\n{'='*70}")
    print(f"❓ {question}")
    print("="*70)

    agent = build_multi_agent()
    initial = {
        "messages": [HumanMessage(content=question)],
        "next_worker": "",
        "final_answer": "",
    }

    final = agent.invoke(initial)
    print(f"\n✅ 最终结论：\n{final['final_answer']}\n")


if __name__ == "__main__":
    # 测试1：业务分析 → Supervisor派research
    ask("恒生电子的AI业务布局怎么样？")

    # 测试2：数字计算 → Supervisor派data
    ask("净利润12.3亿同比增长18.5%，去年同期净利润是多少？")

    # 测试3：总结题 → Supervisor先派research，走完整流水线
    ask("总结一下恒生电子的财务表现和AI战略")
