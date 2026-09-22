"""
LangGraph StateMachine 手写示例
目标：手写一个状态机，理解节点、边、条件路由的原理

类比 Java：
  - State     = Java POJO（一个类定义所有字段）
  - Node      = @Service 方法（一个函数处理一步逻辑）
  - Edge      = 固定的下一步（A 节点执行完，永远去 B 节点）
  - Conditional Edge = switch/if-else（A 节点执行完，根据条件决定去哪）

环境变量：
    export DASHSCOPE_API_KEY="sk-xxx"
    export OPENAI_API_KEY="sk-xxx"
"""

import os
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv


# ============================================================
# 第一步：定义 State（Java 类比：POJO / DTO）
# ============================================================
# TypedDict 就像 Java 的 class，定义这个流程图里有哪些"状态字段"
# 每次节点执行完，都会更新这个 state
load_dotenv()

class AgentState(TypedDict):
    messages: list          # 对话历史（和昨天一样）
    next_action: str        # 下一步动作："use_tool" 或 "answer"
    tool_name: str          # 要调用的工具名
    tool_args: dict         # 工具参数


# ============================================================
# 第二步：定义工具（和昨天一样）
# ============================================================

@tool
def calculator(expression: str) -> str:
    """计算器工具。输入数学表达式，返回计算结果。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_stock_price(stock_name: str) -> str:
    """查询股票价格（模拟数据）"""
    mock_data = {
        "恒生电子": {"price": 42.58, "change": "+2.3%", "pe": 35.2},
        "同花顺": {"price": 128.90, "change": "-1.5%", "pe": 42.8},
        "东方财富": {"price": 18.76, "change": "+0.8%", "pe": 28.5},
    }
    info = mock_data.get(stock_name)
    if info:
        return f"{stock_name}: 现价{info['price']}元, 涨跌{info['change']}, PE={info['pe']}"
    return f"未找到 {stock_name}（仅支持：{list(mock_data.keys())}）"


# 工具注册表（Java 类比：Spring 容器里的 Bean Map）
tools = [calculator, get_stock_price]
tool_map = {t.name: t for t in tools}  # { "calculator": calculator函数, ... }


# ============================================================
# 第三步：定义节点（Java 类比：@Service 方法）
# ============================================================
# 每个节点是一个函数，输入 state，返回更新后的 state

def analyze_question(state: AgentState) -> AgentState:
    """
    节点1：分析用户问题，决定要不要调工具
    Java 类比：Controller 收到请求后，先判断走哪个 Service
    """
    print("\n[节点1] 分析用户问题...")
    
    # 让 LLM 判断：这个问题需要调工具吗？需要哪个工具？
    llm = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    
    # 构造工具描述
    tool_descriptions = "\n".join([
        f"- {t.name}: {t.description}" for t in tools
    ])
    
    prompt = f"""你是一个任务路由器。根据用户问题，判断是否需要调用工具。

可用工具：
{tool_descriptions}

用户问题：{state['messages'][-1].content}

请回答：
1. 是否需要调工具？（yes/no）
2. 如果需要，工具名和参数是什么？（JSON格式）

输出格式：
need_tool: yes/no
tool_name: xxx
tool_args: {{"key": "value"}}
"""
    
    response = llm.invoke(prompt)
    content = response.content
    
    print(f"  LLM 判断结果：\n  {content}")
    
    # 解析 LLM 的输出
    lines = content.strip().split("\n")
    need_tool = "no"
    tool_name = ""
    tool_args = {}
    
    for line in lines:
        if line.startswith("need_tool:"):
            need_tool = line.split(":")[1].strip().lower()
        elif line.startswith("tool_name:"):
            tool_name = line.split(":")[1].strip()
        elif line.startswith("tool_args:"):
            try:
                import json
                tool_args = json.loads(line.split(":", 1)[1].strip())
            except:
                tool_args = {}
    
    # 更新 state
    return {
        "messages": state["messages"],
        "next_action": "use_tool" if need_tool == "yes" else "answer",
        "tool_name": tool_name,
        "tool_args": tool_args,
    }


def execute_tool(state: AgentState) -> AgentState:
    """
    节点2：执行工具调用
    Java 类比：Service 方法执行具体业务逻辑
    """
    tool_name = state["tool_name"]
    tool_args = state["tool_args"]
    
    print(f"\n[节点2] 调用工具：{tool_name}({tool_args})")
    
    # 从工具注册表里找到对应函数，执行
    tool_func = tool_map[tool_name]
    result = tool_func.invoke(tool_args)
    
    print(f"  工具返回：{result}")
    
    # 把工具结果加入 messages
    new_messages = state["messages"] + [AIMessage(content=f"工具调用结果：{result}")]
    
    return {
        "messages": new_messages,
        "next_action": "answer",
        "tool_name": "",
        "tool_args": {},
    }


def generate_answer(state: AgentState) -> AgentState:
    """
    节点3：生成最终回答
    Java 类比：Controller 组装 Response 返回给前端
    """
    print("\n[节点3] 生成最终回答...")
    
    llm = ChatOpenAI(
        model="qwen-plus",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    
    # 如果有工具结果，让 LLM 基于结果回答
    if len(state["messages"]) > 1:
        prompt = "根据以下信息回答用户问题：\n"
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                prompt += f"\n用户问题：{msg.content}\n"
            elif isinstance(msg, AIMessage):
                prompt += f"工具结果：{msg.content}\n"
        prompt += "\n请给出完整、准确的回答。"
    else:
        # 没有工具结果，直接回答
        prompt = state["messages"][-1].content
    
    response = llm.invoke(prompt)
    final_answer = response.content
    
    print(f"  最终回答：{final_answer}")
    
    # 把最终回答加入 messages
    new_messages = state["messages"] + [AIMessage(content=final_answer)]
    
    return {
        "messages": new_messages,
        "next_action": "answer",
        "tool_name": "",
        "tool_args": {},
    }


# ============================================================
# 第四步：定义条件路由（Java 类比：switch/if-else）
# ============================================================

def route_after_analysis(state: AgentState) -> str:
    """
    条件路由：分析完问题后，决定下一步去哪
    Java 类比：
        if (state.next_action == "use_tool") {
            return "execute_tool";
        } else {
            return "generate_answer";
        }
    """
    if state["next_action"] == "use_tool":
        return "execute_tool"
    else:
        return "generate_answer"


# ============================================================
# 第五步：组装状态机（Java 类比：Spring StateMachine 配置）
# ============================================================

def build_agent():
    """
    组装状态机
    Java 类比：
        @Configuration
        public class AgentStateMachine {
            @Bean
            public StateMachine<AgentState> agentMachine() {
                return StateMachineBuilder
                    .withState(AgentState.class)
                    .initial("analyze_question")
                    .end("generate_answer")
                    ...
            }
        }
    """
    
    # 创建状态机
    workflow = StateGraph(AgentState)
    
    # 添加节点（Java 类比：注册 Service Bean）
    workflow.add_node("analyze_question", analyze_question)
    workflow.add_node("execute_tool", execute_tool)
    workflow.add_node("generate_answer", generate_answer)
    
    # 添加边（Java 类比：配置状态转移）
    workflow.add_edge(START, "analyze_question")           # 开始 → 分析问题
    workflow.add_conditional_edges(                        # 分析问题后，根据条件路由
        "analyze_question",
        route_after_analysis,
        {
            "execute_tool": "execute_tool",     # 需要工具 → 执行工具
            "generate_answer": "generate_answer" # 不需要工具 → 直接回答
        }
    )
    workflow.add_edge("execute_tool", "generate_answer")   # 执行完工具 → 生成回答
    workflow.add_edge("generate_answer", END)              # 回答完 → 结束
    
    # 编译成可执行的 Agent（Java 类比：ApplicationContext.refresh()）
    agent = workflow.compile()
    
    return agent


# ============================================================
# 第六步：运行测试
# ============================================================

def run_test(question: str):
    """运行一个测试问题"""
    print(f"\n{'='*70}")
    print(f"❓ 用户问题：{question}")
    print(f"{'='*70}")
    
    agent = build_agent()
    
    # 初始状态
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "next_action": "",
        "tool_name": "",
        "tool_args": {},
    }
    
    # 执行状态机
    final_state = agent.invoke(initial_state)
    
    # 提取最终回答
    final_answer = final_state["messages"][-1].content
    print(f"\n✅ 最终回答：\n{final_answer}\n")
    return final_answer


if __name__ == "__main__":
    # 测试1：需要调工具的问题
    run_test("恒生电子现在多少钱？")
    
    print("\n" + "="*70)
    print("测试2：不需要工具的问题")
    print("="*70)
    
    # 测试2：不需要工具的问题
    run_test("什么是Python？")
    
    print("\n" + "="*70)
    print("测试3：复杂问题（需要计算）")
    print("="*70)
    
    # 测试3：需要计算
    run_test("我有500股同花顺，按当前价算总值多少？")
