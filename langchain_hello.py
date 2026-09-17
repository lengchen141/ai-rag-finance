from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 模型 —— 还是通义千问，langchain-openai兼容dashscope
API_KEY = "sk-ws-H.PMLEIHI.37If.MEYCIQC0gEgk4lT4lSPqWETiidM64fbSfWTCd0wLZC3pwjp6NQIhAIWriH1fEzNIZvOX8yRIaeBjcDSxQGV7Zp4iN9cWqUEL"

llm = ChatOpenAI(
    model="qwen-plus",
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 2. 最简单的调用
print("=== 第1步：裸调用 ===")
resp = llm.invoke("用一句话解释什么是RAG")
print(resp.content)

# 3. Prompt模板 —— 相当于把prompt参数化
print("\n=== 第2步：Prompt模板 ===")
prompt = ChatPromptTemplate.from_template(
    "你是金融分析师，用通俗的话解释{concept}，给个金融场景的例子，100字以内"
)
chain = prompt | llm | StrOutputParser()   # | 是管道符，把三步串起来

print(chain.invoke({"concept": "向量数据库"}))

# 4. 批量调用 —— 一次问多个概念
print("\n=== 第3步：批量调用 ===")
results = chain.batch([
    {"concept": "Embedding"},
    {"concept": "Agent"},
    {"concept": "MCP协议"}
])
for concept, result in zip(["Embedding", "Agent", "MCP"], results):
    print(f"【{concept}】{result}\n")