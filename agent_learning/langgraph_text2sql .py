"""
Agent + 数据库：自然语言查 SQL（Text-to-SQL）

流程：
    用户大白话
       ↓ Agent（LLM）
    生成 SQL → 执行 SQLite → 拿到结果
       ↓ Agent（LLM）
    把结果翻译成人话

Java 类比：
    以前报表 = Controller 写死 SQL（JDBC 执行）
    现在     = LLM 根据用户的话动态拼 SQL，再走 JDBC 执行
    Text-to-SQL 是企业落地最多的 AI 形态之一，面试高频

特点：用 Python 自带 sqlite3 建内存库，无需安装数据库，直接能跑

.env 文件：
    OPENAI_API_KEY=sk-xxx
    DASHSCOPE_API_KEY=sk-xxx
"""

import os
import sqlite3
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 固定动作：加载 .env
load_dotenv()


# ============================================================
# 第一步：建一个模拟的股票交易数据库（SQLite 内存库）
# ============================================================
def init_database():
    """建表 + 插入模拟数据，返回连接"""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cur = conn.cursor()

    # 股票基础信息表（Java 类比：stock 表）
    cur.execute("""
        CREATE TABLE stock (
            code TEXT PRIMARY KEY,   -- 股票代码
            name TEXT,               -- 股票名称
            industry TEXT            -- 所属行业
        )
    """)

    # 每日交易行情表
    cur.execute("""
        CREATE TABLE quote (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_code TEXT,         -- 股票代码
            trade_date TEXT,         -- 交易日期
            close REAL,              -- 收盘价
            volume REAL,             -- 成交量(万股)
            turnover REAL            -- 成交额(万元)
        )
    """)

    # 插入股票数据
    stocks = [
        ("600570", "恒生电子", "金融IT"),
        ("300033", "同花顺", "金融IT"),
        ("300059", "东方财富", "金融IT"),
        ("601318", "中国平安", "保险"),
    ]
    cur.executemany("INSERT INTO stock VALUES (?,?,?)", stocks)

    # 插入行情数据（两天，便于演示"上个月/对比"）
    quotes = [
        # 9月18日
        ("600570", "2026-09-18", 42.58, 3200, 136200),
        ("300033", "2026-09-18", 128.90, 2100, 270690),
        ("300059", "2026-09-18", 18.76, 15000, 281400),
        ("601318", "2026-09-18", 52.30, 9800, 512540),
        # 9月19日
        ("600570", "2026-09-19", 43.20, 3500, 151200),
        ("300033", "2026-09-19", 127.10, 1900, 241490),
        ("300059", "2026-09-19", 19.02, 14200, 270084),
        ("601318", "2026-09-19", 52.80, 10100, 533280),
    ]
    cur.executemany(
        "INSERT INTO quote (stock_code, trade_date, close, volume, turnover) VALUES (?,?,?,?,?)",
        quotes,
    )
    conn.commit()
    return conn


db_conn = init_database()


# 数据库表结构说明（关键！要告诉 LLM 有哪些表、字段啥意思）
DB_SCHEMA = """
数据库类型：SQLite
表结构：

1) stock（股票基础信息表）
   - code TEXT       股票代码，如 '600570'
   - name TEXT       股票名称，如 '恒生电子'
   - industry TEXT   所属行业，如 '金融IT'

2) quote（每日行情表）
   - stock_code TEXT 股票代码（关联 stock.code）
   - trade_date TEXT 交易日期，格式 'YYYY-MM-DD'
   - close REAL      收盘价（元）
   - volume REAL     成交量（万股）
   - turnover REAL   成交额（万元）
"""


# ============================================================
# 第二步：定义"查数据库"工具
# ============================================================
@tool
def query_database(sql: str) -> str:
    """执行 SQL 查询股票数据库。当用户询问股票价格、成交额、成交量、
    行业、排名、对比等数据类问题时使用。只能用 SELECT 查询。
    输入：一条合法的 SQLite SELECT 语句。
    """
    sql = sql.strip().rstrip(";")

    # 安全护栏：只允许 SELECT（防止 LLM 生成删库语句）
    if not sql.upper().startswith("SELECT"):
        return "安全限制：只允许执行 SELECT 查询。"

    try:
        cur = db_conn.cursor()
        cur.execute(sql)
        columns = [d[0] for d in cur.description]  # 列名
        rows = cur.fetchall()
        if not rows:
            return "查询成功，但没有符合条件的数据。"

        # 把结果拼成文本返回给 Agent
        result_lines = [" | ".join(columns)]
        for row in rows:
            result_lines.append(" | ".join(str(v) for v in row))
        return "\n".join(result_lines)
    except Exception as e:
        return f"SQL 执行出错：{e}"


@tool
def calculator(expression: str) -> str:
    """数学计算器。输入数学表达式，返回计算结果。"""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"calculator error: {e}"


# ============================================================
# 第三步：创建 Agent（数据库 + 计算器）
# ============================================================
llm = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

# 把表结构放进 system prompt，让 LLM 知道怎么写 SQL
system_prompt = f"""你是一个金融数据助手。你可以通过 query_database 工具查询数据库来回答用户问题。

{DB_SCHEMA}

要求：
1. 数据类问题，先用 query_database 查出数据，再组织语言回答。
2. 根据表字段写正确的 SQLite 语句，需要时用 JOIN 关联两张表。
3. 涉及计算（如涨幅、百分比）可以用 calculator 工具。
4. 回答时给出具体数字，必要时说明数据日期。
"""

tools = [query_database, calculator]
agent = create_react_agent(llm, tools, prompt=system_prompt)


# ============================================================
# 第四步：观察 Agent 怎么写 SQL
# ============================================================
def ask(question: str):
    print(f"\n{'='*60}")
    print(f"❓ {question}")
    print("=" * 60)

    result = agent.invoke({"messages": [("user", question)]})

    print("\n📋 Agent 执行过程:")
    for msg in result["messages"]:
        role = msg.__class__.__name__
        tool_calls = getattr(msg, "tool_calls", [])
        if role == "AIMessage" and tool_calls:
            for tc in tool_calls:
                print(f"  → 调用 {tc['name']}：")
                # SQL 单独打印，方便看
                if tc["name"] == "query_database":
                    print(f"      SQL: {tc['args'].get('sql')}")
                else:
                    print(f"      参数: {tc['args']}")
        elif role == "ToolMessage":
            print(f"  ← 返回:\n{msg.content}")

    print(f"\n💬 最终回答：\n{result['messages'][-1].content}")


if __name__ == "__main__":
    # 1. 简单查询：单只股票价格
    ask("恒生电子9月19日的收盘价是多少？")

    # 2. 聚合 + 排序：成交额最大的股票（需要 JOIN 拿股票名）
    ask("9月19日成交额最大的股票是哪只？")

    # 3. 计算涨幅：需要查两天收盘价再算
    ask("中国平安9月18日到19日涨了百分之多少？")

    # 4. 分组统计：每个行业的情况
    ask("金融IT行业有哪些公司？")
