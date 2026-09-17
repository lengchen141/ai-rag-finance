"""
W3 Day3: RAG封装成FastAPI服务 —— 项目①原型
对标Java类比：
  - FastAPI ≈ Spring Boot Controller
  - Pydantic模型 ≈ Java DTO/Request对象
  - uvicorn启动 ≈ java -jar xxx.jar
  - GET /ask ≈ @GetMapping("/ask")
"""
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
from uuid import uuid4
from typing import Optional
from dotenv import load_dotenv
# ============ 1. 配置 ============
# key换成你自己的
load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

PDF_PATH = "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"   # 和昨天同一个PDF
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

pdf_files = []  # 全局：当前已加载的PDF列表

# 对话历史存储：{session_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
chat_histories = {}

# ============ 2. Pydantic请求模型（Java类比：DTO）============
class AskRequest(BaseModel):
    """提问请求，支持GET参数和POST JSON两种用法"""
    question: str  # 用户问题
    k: int = 3     # 检索Top-K，默认3

class AskResponse(BaseModel):
    """回答响应"""
    question: str
    answer: str
    sources: list[str]  # 引用的资料来源（方便调试和溯源）

# ============ 3. RAG初始化（服务启动时一次性加载）============
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

qa_chain = None  # 全局变量，启动时初始化

def init_rag():
    """加载所有PDF、切块、建FAISS库"""
    global qa_chain, pdf_files
    
    # 扫描uploads目录下的所有PDF
    pdf_files = list(UPLOAD_DIR.glob("*.pdf"))
    if not pdf_files:
        # 如果没有上传的PDF，用默认的
        default_pdf = Path("research_report.pdf")
        if default_pdf.exists():
            pdf_files = [default_pdf]
    
    # ★ 新增：检查是否有PDF
    if not pdf_files:
        print("⚠️ 未找到任何PDF文件，请先上传或放置 research_report.pdf 到项目根目录")
        qa_chain = None
        return
    
    print(f" 找到 {len(pdf_files)} 个PDF文件")
    
    all_docs = []
    for pdf_path in pdf_files:
        print(f"  📄 加载: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        all_docs.extend(docs)
    
    # ★ 新增：检查是否有内容
    if not all_docs:
        print("⚠️ PDF解析后无内容，请检查PDF是否为扫描件")
        qa_chain = None
        return
    
    print(f"✂️ 切块中...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "，", " "],
    )
    chunks = splitter.split_documents(all_docs)
    
    print(" 向量化中（DashScope embedding）...")
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    db = FAISS.from_documents(chunks, embeddings)
    
    print(" 初始化RetrievalQA Chain...")
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )
    print(f"✅ RAG服务初始化完成（共 {len(chunks)} 个chunks）\n")

def rebuild_rag():
    """上传新PDF后重建RAG库"""
    global qa_chain
    if qa_chain is None:
        print(" 首次初始化RAG...")
    else:
        print(" 重新构建RAG库...")
    init_rag()

# ============ 4. 服务启动/关闭生命周期（Java类比：@PostConstruct）============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时加载RAG，关闭时清理"""
    init_rag()
    yield
    print("服务已关闭")

# ============ 5. FastAPI应用（Java类比：@SpringBootApplication）============
app = FastAPI(
    title="金融研报RAG问答",
    description="基于真实研报PDF的RAG问答服务",
    version="0.1.0",
    lifespan=lifespan,
)


# 静态文件服务：让前端HTML可以通过 /rag_frontend.html 访问
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/rag_frontend.html")
async def frontend():
    """直接返回前端HTML页面"""
    return FileResponse("rag_frontend.html")

@app.get("/")
async def root():
    """根路径重定向到前端页面"""
    return FileResponse("rag_frontend.html")


# ============ 6. 接口定义 ============

class AskRequest(BaseModel):
    """提问请求"""
    question: str
    k: int = 3
    session_id: Optional[str] = None  # 新增：会话ID，不传则自动创建

class AskResponse(BaseModel):
    """回答响应"""
    question: str
    answer: str
    session_id: str  # 新增：返回会话ID
    sources: list[str]
    history_count: int  # 新增：当前对话轮数

@app.get("/ask", response_model=AskResponse)
async def ask_question(
    question: str,
    k: int = 3,
    session_id: Optional[str] = None
):
    """
    GET接口：提问并获取基于研报的回答（支持多轮对话）
    用法：http://localhost:8002/ask?question=xxx&session_id=xxx
    """
    if not qa_chain:
        raise HTTPException(status_code=503, detail="RAG服务未就绪")
    
    # 如果没有session_id，创建新的
    if not session_id:
        session_id = str(uuid4())
    
    # 获取或初始化对话历史
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    
    history = chat_histories[session_id]
    
    # 构建带历史的prompt
    context_prompt = "以下是之前的对话历史：\n"
    if history:
        for msg in history[-6:]:  # 只取最近6条，避免prompt太长
            role = "用户" if msg["role"] == "user" else "助手"
            context_prompt += f"{role}：{msg['content']}\n"
        context_prompt += "\n请基于以上对话历史和参考资料回答新问题。\n"
    else:
        context_prompt += "请基于参考资料回答用户问题。\n"
    
    # 检索参考资料
    result = qa_chain.invoke({"query": question})
    sources = [doc.page_content[:150] for doc in result.get("source_documents", [])]
    
    # 把历史记录拼进prompt（这里简化处理，实际可以用ChatPromptTemplate）
    # 注意：RetrievalQA默认不直接支持history，需要手动拼
    full_question = context_prompt + f"新问题：{question}"
    result = qa_chain.invoke({"query": full_question})
    
    answer = result["result"]
    
    # 保存对话历史
    history.append({"role": "user", "content": question})
    history.append({"role": "assistant", "content": answer})
    
    return AskResponse(
        question=question,
        answer=answer,
        session_id=session_id,
        sources=sources,
        history_count=len(history) // 2,  # 轮数 = 消息数 / 2
    )

@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """清空指定会话的历史"""
    if session_id in chat_histories:
        del chat_histories[session_id]
        return {"message": "会话已清空"}
    raise HTTPException(status_code=404, detail="会话不存在")

# CORS：允许前端跨域访问（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    上传PDF文件，自动重建RAG库
    用法：POST /upload，body为文件
    """
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="只支持PDF文件")
    
    # 保存到uploads目录
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # 重建RAG库
    rebuild_rag()
    
    return {
        "message": f"上传成功：{file.filename}",
        "total_pdfs": len(pdf_files),
    }

@app.get("/documents")
async def list_documents():
    """列出当前已加载的所有PDF"""
    return {
        "documents": [f.name for f in pdf_files],
        "count": len(pdf_files),
    }

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """删除指定PDF并重建RAG库"""
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    
    file_path.unlink()
    rebuild_rag()
    
    return {"message": f"已删除：{filename}"}

@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok", "rag_ready": qa_chain is not None}

# ============ 7. 启动入口（Java类比：main方法）============
if __name__ == "__main__":
    uvicorn.run(
        "rag_api_server:app",
        host="0.0.0.0",
        port=8002,  # 注意：和之前8000/8001区分
        reload=True,  # 代码修改后自动重载（开发模式）
    )