"""
W3 Day2: LangChain RetrievalQA Chain 封装RAG
对标Java类比：
  - RetrievalQA ≈ Spring的JpaRepository，你把SQL细节交给它，只暴露save()/find()
  - Document Loader ≈ Java的ResourceLoader，统一PDF/DOCX/TXT等格式
  - Chain ≈ Java Stream管道，数据从Loader → Splitter → VectorStore → QA Chain 一路流过去
"""
import os

# ============ 1. 配置 ============
API_KEY = "sk-ws-H.PMLEIHI.37If.MEYCIQC0gEgk4lT4lSPqWETiidM64fbSfWTCd0wLZC3pwjp6NQIhAIWriH1fEzNIZvOX8yRIaeBjcDSxQGV7Zp4iN9cWqUEL"

os.environ["DASHSCOPE_API_KEY"] = API_KEY
os.environ["OPENAI_API_KEY"] = API_KEY

# ============ 2. Document Loading（替代昨天的PdfReader）============
from langchain_community.document_loaders import PyPDFLoader

PDF_PATH = "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"  # 和昨天同一个PDF

def load_pdf(pdf_path: str):
    """LangChain Document Loader，直接返回带元数据的文档对象"""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"PDF解析完成，共 {len(docs)} 页（每页一个Document对象）")
    print(f"示例文档元数据: {docs[0].metadata}")
    print(f"示例文档内容前100字: {docs[0].page_content[:100]}...")
    return docs

# ============ 3. 切块（和昨天一样，但输入变成Document列表）============
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_docs(docs, chunk_size: int = 300, chunk_overlap: int = 50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "，", " "],
    )
    chunks = splitter.split_documents(docs)  # 注意：这里是split_documents，不是create_documents
    print(f"切块后数量: {len(chunks)}")
    return chunks

# ============ 4. 向量化 + FAISS（和昨天一样）============
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS

def build_vector_store(chunks):
    embeddings = DashScopeEmbeddings(
        model="text-embedding-v3",
        dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
    )
    print("正在调DashScope embedding API...")
    db = FAISS.from_documents(chunks, embeddings)
    print(f"FAISS库建好，共 {db.index.ntotal} 个向量")
    return db

# ============ 5.  RetrievalQA Chain（核心封装）============
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

def build_retrieval_qa(db):
    """
    用RetrievalQA封装整个RAG流程
    一行代码替代昨天的：检索 + 拼prompt + 调LLM
    """
    llm = ChatOpenAI(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        temperature=0,
    )
    
    # 关键一行：把retriever（FAISS）+ LLM 封装成Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "stuff"=把检索结果全塞进prompt（最常用）
        retriever=db.as_retriever(search_kwargs={"k": 3}),  # 指定Top-3
        # 可选：自定义prompt（和昨天手搓版一样）
        # chain_type_kwargs={"prompt": custom_prompt}
    )
    return qa_chain

# ============ 6. 主流程 ============
def main():
    print("=" * 60)
    print("LangChain RetrievalQA Chain 封装版RAG")
    print("=" * 60)
    
    from pathlib import Path
    if not Path(PDF_PATH).exists():
        print(f"\n⚠️ PDF文件不存在: {PDF_PATH}")
        return
    
    # 全流程串联（和昨天一样的4步）
    docs = load_pdf(PDF_PATH)
    chunks = split_docs(docs)
    db = build_vector_store(chunks)
    qa_chain = build_retrieval_qa(db)
    
    # 跑查询
    questions = [
        "公司AI业务布局和进展？",
        "营收和利润表现如何？",
        "公司在行业中的竞争优势是什么？",
    ]
    
    for q in questions:
        print(f"\n❓ 用户提问: {q}")
        
        #  核心：一行调用Chain
        result = qa_chain.invoke({"query": q})
        
        print(f"\n[LLM回答]\n{result['result']}")
        print("\n" + "-" * 60)

if __name__ == "__main__":
    main()