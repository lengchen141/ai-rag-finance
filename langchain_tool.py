import json
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

# ========== 1. 定义工具（相当于 Java 里一个普通方法，加个 @tool 注解）==========

@tool
def get_stock_price(stock_code: str) -> str:
    """获取A股某只股票的实时价格。当用户问"股价"、"价格"、"行情"时调用。"""
    # 真实项目这里调 tushare/akshare 等行情接口，这里用 mock 数据
    prices = {
        "600519": {"name": "贵州茅台", "price": 1820.5, "change": +1.2},
        "300750": {"name": "宁德时代", "price": 210.3, "change": -0.5},
        "002594": {"name": "比亚迪", "price": 245.8, "change": +2.1},
    }
    data = prices.get(stock_code)
    if data:
        return f"{data['name']}({stock_code}) 当前价:{data['price']} 涨跌:{data['change']}%"
    return f"未找到股票 {stock_code}"

@tool
def exchange_rate(usd_amount: float) -> str:
    """将美元换算成人民币，当用户问汇率、换汇时调用。"""
    rate = 7.25  # mock
    return f"{usd_amount} 美元 ≈ {usd_amount * rate:.2f} 人民币（汇率7.25）"

# ========== 2. 把工具注册给大模型 ==========
API_KEY = "sk-ws-H.PMLEIHI.37If.MEYCIQC0gEgk4lT4lSPqWETiidM64fbSfWTCd0wLZC3pwjp6NQIhAIWriH1fEzNIZvOX8yRIaeBjcDSxQGV7Zp4iN9cWqUEL"

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# bind_tools 把工具 schema 注入到模型里，模型"知道"能调用什么函数
llm_with_tools = llm.bind_tools([get_stock_price, exchange_rate])

# ========== 3. 测试：看模型会不会"想"调用工具 ==========

print("=== 测试1：问股价（应该触发 get_stock_price）===")
resp1 = llm_with_tools.invoke("贵州茅台现在多少钱？")
print(f"模型回复: {resp1}")
print(f"tool_calls: {resp1.tool_calls}\n")  # ← 这里能看到模型返回的"函数调用请求"

print("=== 测试2：问普通问题（不应该触发工具）===")
resp2 = llm_with_tools.invoke("什么是Java？")
print(f"模型回复: {resp2}\n")

print("=== 测试3：问汇率（应该触发 exchange_rate）===")
resp3 = llm_with_tools.invoke("1000美元换人民币多少？")
print(f"tool_calls: {resp3.tool_calls}\n")

# ========== 4. 完整流程：让模型真的执行工具并拿到结果 ==========
# 这一步才是 Agent 真正的循环：用户问 → 模型决定调哪个工具 → 我们执行工具 → 把结果喂回模型 → 模型生成最终回答

def agent_loop(question: str):
    """手动实现一遍 ReAct 循环，让你看清楚 Agent 是怎么工作的"""
    print(f"\n用户问: {question}")
    
    # 第一步：让模型决定
    messages = [HumanMessage(content=question)]
    resp = llm_with_tools.invoke(messages)
    messages.append(resp)
    
    # 第二步：如果模型没调用工具，直接返回
    if not resp.tool_calls:
        print(f"AI: {resp.content}")
        return
    
    # 第三步：执行工具（这里用 mock 分发，真实项目用 ToolNode）
    tools_map = {
        "get_stock_price": get_stock_price,
        "exchange_rate": exchange_rate
    }
    
    for tc in resp.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        print(f"  → 模型决定调用: {tool_name}({tool_args})")
        
        # 执行工具拿到结果
        tool_result = tools_map[tool_name].invoke(tool_args)
        print(f"  → 工具返回: {tool_result}")
        
        # 把工具结果作为 ToolMessage 加回对话
        messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))
    
    # 第四步：带着工具结果再问一次模型，生成最终回答
    final = llm_with_tools.invoke(messages)
    print(f"AI最终回答: {final.content}")

# ========== 5. 跑几个完整循环 ==========

agent_loop("贵州茅台现在多少钱？帮我看看")
agent_loop("我手里5000美元，想换成人民币，多少？")
agent_loop("什么是Agent？")  # 这个不会调工具