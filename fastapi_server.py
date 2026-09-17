from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

# ========== FastAPI 初始化 ==========
# 相当于 Java 的 SpringApplication.run() 但简洁得多
app = FastAPI(title="金融AI助手API", version="0.1.0")

# ========== 请求/响应模型（相当于 Java 的 DTO / Request/Response 类）==========
# Pydantic BaseModel 相当于 Java 的 record，自带校验
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []    # 对话历史，可选

class ChatResponse(BaseModel):
    reply: str
    model: str = "qwen-plus"

# ========== 初始化 AI 客户端 ==========
API_KEY = "sk-ws-H.PMLEIHI.37If.MEYCIQC0gEgk4lT4lSPqWETiidM64fbSfWTCd0wLZC3pwjp6NQIhAIWriH1fEzNIZvOX8yRIaeBjcDSxQGV7Zp4iN9cWqUEL"
client = OpenAI(
    api_key=API_KEY,
    base_url="https://ws-ndaimedbdbwxttvo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

# ========== 路由定义 ==========

# 健康检查 - 相当于 @GetMapping("/health")
@app.get("/")
def root():
    return {"status": "ok", "service": "金融AI助手", "version": "0.1.0"}

# AI 对话接口 - 相当于 @PostMapping("/chat")
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    messages = [
        {"role": "system", "content": "你是一个专业的金融分析师，回答简洁专业。"}
    ]
    # 把历史对话拼进去
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="qwen-plus",
        messages=messages,
    )

    reply = response.choices[0].message.content
    return ChatResponse(reply=reply)

# ========== 启动 ==========
# 相当于 java -jar xxx.jar 但不用打包
# 运行命令: uvicorn fastapi_server:app --reload --port 8000