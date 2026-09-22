"""
LangGraph Agent 最小可跑示例
目标：让 LLM 自己决定"调计算器"还是"查股票价格"

安装：
    pip install langgraph langchain-openai langchain-community

类比 Java：
  - Tool       ≈ @Service Bean（一个方法就是一个工具）
  - Agent      ≈ @RestController 里动态决策的 Controller
  - State      ≈ Spring StateMachine 的状态机
  - ReAct      ≈ AOP 拦截器链：Thought → Action → Observation → ...

环境变量：
    export DASHSCOPE_API_KEY="sk-xxx"      # Embedding 用
    export OPENAI_API_KEY="sk-xxx"         # LLM 用（同值，两个通道）
"""

import os
from typing import Annotated, TypedDict

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv


# ============================================================
# 第一步：定义工具（Java 类比：@Service Bean）
# ============================================================
# @tool 装饰器 = 把这个函数注册成一个 Agent 可调用的工具
# LLM 能看到函数名 + 注释 + 参数类型，然后决定是否调用

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """计算器工具。输入数学表达式，返回计算结果。
    例如："123 * 456" 或 "100 / 7 + 32"
    """
    try:
        # 注意：生产环境不能直接 eval，这里为了教学
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_stock_price(stock_name: str) -> str:
    """查询股票的当前价格（模拟数据，真实场景调 API）
    输入：股票名称，如 "恒生电子" 或 "同花顺"
    """
    # 真实场景：调东方财富 / 同花顺 / 雅虎财经 API
    mock_data = {
        "恒生电子": {"price": 42.58, "change": "+2.3%", "pe": 35.2},
        "同花顺": {"price": 128.90, "change": "-1.5%", "pe": 42.8},
        "东方财富": {"price": 18.76, "change": "+0.8%", "pe": 28.5},
        "中国平安": {"price": 52.30, "change": "-0.3%", "pe": 9.2},
    }
    info = mock_data.get(stock_name)
    if info:
        return f"{stock_name}: 现价{info['price']}元, 涨跌{info['change']}, PE={info['pe']}"
    return f"未找到 {stock_name} 的行情数据（仅支持：{list(mock_data.keys())}）"


# ============================================================
# 第二步：创建 Agent（Java 类比：注册 Controller）
# ============================================================
# create_react_agent 是 LangGraph 的快捷方式，等价于：
#   new StateGraph(AgentState)
#     .addNode("agent", llm)
#     .addNode("tools", ToolNode)
#     .addEdge(...)
#     .compile()

# 工具清单（Java 类比：@Bean 注册到 Spring 容器）
tools = [calculator, get_stock_price]

# LLM 实例（和昨天 RAG 一样，qwen-plus）
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

# 一行创建 ReAct Agent
agent = create_react_agent(llm, tools)


# ============================================================
# 第三步：运行 Agent（Java 类比：HTTP 请求打进来）
# ============================================================
def run_agent(question: str):
    """跑一次 Agent，打印完整 ReAct 过程"""
    print(f"\n{'='*60}")
    print(f"❓ 问题：{question}")
    print(f"{'='*60}")

    # agent.invoke() 返回完整执行历史
    result = agent.invoke({"messages": [("user", question)]})

    # 打印过程（类似 Java 看 Spring 的 DEBUG 日志）
    print("\n📋 执行过程（ReAct 循环）:")
    for i, msg in enumerate(result["messages"]):
        role = msg.__class__.__name__
        content = msg.content if hasattr(msg, "content") else str(msg)
        tool_calls = getattr(msg, "tool_calls", [])

        if role == "HumanMessage":
            print(f"  [{i}] 👤 User: {content[:100]}")
        elif role == "AIMessage":
            if tool_calls:
                for tc in tool_calls:
                    print(f"  [{i}] 🤖 LLM 决定调用: {tc['name']}({tc['args']})")
            else:
                print(f"  [{i}] 🤖 LLM 最终回答: {content[:200]}")
        elif role == "ToolMessage":
            print(f"  [{i}] 🔧 工具返回: {content[:150]}")

    # 最终答案
    final_answer = result["messages"][-1].content
    print(f"\n✅ 最终回答:\n{final_answer}\n")
    return final_answer


# ============================================================
# 第四步：测试三个问题，看 Agent 怎么决策
# ============================================================
if __name__ == "__main__":
    # 测试 1：纯计算（Agent 应该调 calculator）
    run_agent("帮我算一下 12345 * 6789 等于多少")

    # 测试 2：查股票（Agent 应该调 get_stock_price）
    run_agent("恒生电子现在多少钱？同花顺呢？")

    # 测试 3：混合问题（Agent 应该先查股票再算涨幅）
    run_agent("我有 1000 股平安银行，按当前价算总值多少？")

    # 测试 4：超出能力的问题（Agent 会诚实回答不知道）
    run_agent("明天恒生电子股价会涨到多少？")
