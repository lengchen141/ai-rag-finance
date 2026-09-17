"""
W3 Day1: Embedding + FAISS 向量检索 —— RAG的命门
对标Java类比：
  - Embedding ≈ 把文本hash成固定维度的float[]，只不过这里用神经网络"语义理解"
  - FAISS ≈ Java HashMap的"语义版"，key不是字符串而是向量，按余弦相似度检索
  - RAG = Retrieval(FAISS检索) + Augmented(拼进prompt) + Generation(LLM回答)
"""
import os
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ============ 1. Embedding配置（通义DashScope兼容模式）============
# key换成你自己的
API_KEY = "sk-ws-H.PMLEIHI.37If.MEYCIQC0gEgk4lT4lSPqWETiidM64fbSfWTCd0wLZC3pwjp6NQIhAIWriH1fEzNIZvOX8yRIaeBjcDSxQGV7Zp4iN9cWqUEL"

os.environ["DASHSCOPE_API_KEY"] = API_KEY
# OpenAI兼容协议用（ChatOpenAI调LLM用）
os.environ["OPENAI_API_KEY"] = API_KEY
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.environ["DASHSCOPE_API_KEY"],
)

# ============ 2. 模拟研报片段（RAG知识库）============
research_docs = [
    "恒生电子2025年净利润12.3亿元，同比增长18.5%。AI大模型业务收入占比从5%提升至15%，成为第二增长曲线。",
    "恒生电子发布LightGPT金融大模型，在研报解读、合规审查、智能客服三大场景实现商业化落地。",
    "公司合同负债环比增长23%，主要源于券商和基金公司的AI智能体订单增长，反映金融机构对AI Agent需求旺盛。",
    "同花顺2025年营收89亿元，其中AI增值服务贡献占比达42%，iFinD智能终端用户付费率提升11个百分点。",
    "蚂蚁集团推出百灵大模型金融专版，通过Agent编排能力支持银行、保险客户快速搭建信贷审批、理赔自动化流程。",
    "阿里云通义千问开源Qwen2.5系列，支持100万token长上下文，金融客户可用于年报/研报长文档理解。",
    "招行2025半年报显示，智能客服Token消耗同比+78%，智能投顾日均处理咨询量达12万条，AI渗透率行业领先。",
    "平安银行上线140个运营审核类智能体，贷款审批时效从3天缩短至4小时，风险识别准确率提升34%。",
]

print(f"原始文档数: {len(research_docs)}")
print(f"单文档长度: 平均 {sum(len(d) for d in research_docs)//len(research_docs)} 字\n")

# ============ 3. 文本切块（Chunking）============
# Java类比：类似StringTokenizer，但按语义边界（句号/段落）切，更智能
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,      # 每块最大120字符（金融短句子足够一个语义单元）
    chunk_overlap=20,    # 相邻块重叠20字符，防止边界截断丢失语义
    separators=["。", "，", " "],  # 中文语境切分符
)

chunks = splitter.create_documents(research_docs)
print(f"切块后数量: {len(chunks)}")
print(f"示例chunk: {chunks[0].page_content[:80]}...\n")

# ============ 4. 向量化 + 建FAISS库（核心一步）============
# Java类比：HashMap<embedding_vector, document>，但按余弦相似度检索
print("正在调通义embedding API，向量化所有chunk...")
db = FAISS.from_documents(chunks, embeddings)
print(f"FAISS库建好，共 {db.index.ntotal} 个向量\n")

# ============ 5. 相似度检索（检索Top3）============
queries = [
    "恒生电子AI业务表现如何？",
    "银行AI智能体有哪些落地案例？",
    "同花顺的AI营收是多少？",
]

for q in queries:
    print(f"❓ 查询: {q}")
    results = db.similarity_search_with_score(q, k=3)
    for i, (doc, score) in enumerate(results):
        # 分数越小越相似（余弦距离）
        print(f"  [{i+1}] (距离{score:.4f}) {doc.page_content[:60]}...")
    print()

# ============ 6. RAG最小闭环：检索 + 拼prompt + 问LLM============
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""你是金融研报分析助手，仅基于以下参考资料回答问题。
如果参考资料中没有相关信息，明确说"资料中未提及"，不要编造。

参考资料：
{context}

用户问题：{question}

回答：""")

print("=" * 60)
print("RAG完整链路演示")
print("=" * 60)
question = "恒生电子AI业务增长情况如何？"
print(f"\n用户提问：{question}")

# Step1: 检索最相关的3个chunk
retrieved = db.similarity_search(question, k=3)
context = "\n".join([f"- {d.page_content}" for d in retrieved])
print(f"\n[检索到的参考资料]\n{context}\n")

# Step2: 拼prompt + 调LLM
chain = prompt | llm
answer = chain.invoke({"context": context, "question": question})
print(f"[LLM回答]\n{answer.content}")