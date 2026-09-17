"""
W3 Day2: 真实PDF研报 → RAG
对标Java类比：
  - PDF解析 ≈ Java的Apache PDFBox库，把PDF页→纯文本
  - 真实RAG vs 模拟RAG：从"8段手写字"变成"研报全文切出200+个chunk"
  - chunk质量决定RAG效果上限（garbage in, garbage out）
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# ============ 1. 配置 ============
# key换成你自己的
load_dotenv()

os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# ============ 2. PDF解析 ============
from pypdf import PdfReader

PDF_PATH = "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"  # 改成你实际的PDF路径，例如 "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"

def pdf_to_text(pdf_path: str) -> str:
    """把PDF所有页面文字提取出来，拼成一个大字符串"""
    reader = PdfReader(pdf_path)
    print(f"PDF共 {len(reader.pages)} 页")
    
    # 对标Java: StringBuilder 拼接
    text_parts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:  # 有些页是图片无文字，跳过
            text_parts.append(page_text)
    
    full_text = "\n".join(text_parts)
    print(f"提取全文长度: {len(full_text)} 字符")
    return full_text

# ============ 3. 文本切块 ============
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(text: str, chunk_size: int = 300, chunk_overlap: int = 50):
    """
    把长文本切成小块
    chunk_size=300 比模拟demo大，因为研报段落一般较长
    chunk_overlap=50 保留50字重叠防边界截断
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "],  # 按段落>行>句>词递归切
    )
    chunks = splitter.create_documents([text])
    print(f"切块后数量: {len(chunks)}")
    print(f"示例chunk[0]: {chunks[0].page_content[:100]}...")
    print(f"示例chunk[-1]: {chunks[-1].page_content[:100]}...")
    return chunks

# ============ 4. 向量化 + FAISS ============
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(chunks):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    print("正在调DashScope embedding API...")
    db = FAISS.from_documents(chunks, embeddings)
    print(f"FAISS库建好，共 {db.index.ntotal} 个向量\n")
    return db

# ============ 5. RAG检索 + 生成 ============
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

def rag_query(db, question: str, k: int = 3) -> str:
    """完整RAG流程：检索 → 拼prompt → LLM回答"""
    
    # Step1: 检索Top-K
    retrieved = db.similarity_search(question, k=k)
    context = "\n".join([f"- {d.page_content}" for d in retrieved])
    
    print(f"\n❓ 用户提问: {question}")
    print(f"\n[检索到的参考资料]")
    for i, doc in enumerate(retrieved):
        print(f"  [{i+1}] {doc.page_content[:120]}...")
    
    # Step2: 拼prompt + 调LLM
    prompt = ChatPromptTemplate.from_template("""你是金融研报分析助手，仅基于以下参考资料回答问题。
如果参考资料中没有相关信息，明确说"资料中未提及"，不要编造。

参考资料：
{context}

用户问题：{question}

回答：""")
    
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    
    chain = prompt | llm
    answer = chain.invoke({"context": context, "question": question})
    
    print(f"\n[LLM回答]\n{answer.content}")
    return answer.content

# ============ 6. 主流程 ============
def main():
    print("=" * 60)
    print("真实研报PDF → RAG")
    print("=" * 60)
    
    # 检查PDF文件是否存在
    if not Path(PDF_PATH).exists():
        print(f"\n⚠️ PDF文件不存在: {PDF_PATH}")
        print("请下载一份研报PDF放到当前目录，改PDF_PATH变量指向它")
        print("推荐来源：")
        print("  - 恒生电子/同花顺/招行 最新半年报或研报")
        print("  - 东方财富/同花顺/巨潮资讯网都能下载")
        return
    
    # 全流程串联
    full_text = pdf_to_text(PDF_PATH)
    chunks = split_text(full_text)
    db = build_vector_store(chunks)
    
    # 跑3个查询
    questions = [
        "公司AI业务布局和进展？",
        "营收和利润表现如何？",
        "公司在行业中的竞争优势是什么？",
    ]
    
    for q in questions:
        rag_query(db, q)
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()