from fastapi import FastAPI
from pydantic import BaseModel
from openai import AsyncOpenAI   # 注意：异步客户端
import os
from dotenv import load_dotenv
app = FastAPI(title="金融AI助手API-异步版", version="0.2.0")

class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
# 异步客户端
load_dotenv()

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://ws-ndaimedbdbwxttvo.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)

@app.get("/")
async def root():                          # async def
    return {"status": "ok", "version": "0.2.0-async"}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):          # async def
    messages = [
        {"role": "system", "content": "你是一个专业的金融分析师，回答简洁专业。"}
    ]
    messages.extend(req.history)
    messages.append({"role": "user", "content": req.message})

    response = await client.chat.completions.create(   # await
        model="qwen-plus",
        messages=messages,
    )
    return ChatResponse(reply=response.choices[0].message.content)