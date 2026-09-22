"""
混合检索 + Rerank：BM25 关键词检索 × 向量语义检索 → RRF融合 → 重排序

解决纯向量检索的软肋：
    - 股票代码、专有名词、数字 → BM25 精确匹配擅长
    - 同义词、语义改写         → 向量检索擅长
    - 两者融合，召回更全；Rerank 再把最相关的顶到前面

Java 类比：
    两个 Service 各查各的（关键词库 / 向量库）
    FusionService 按 RRF 规则合并去重，再用精排服务重排

安装：
    pip install jieba rank_bm25 faiss-cpu langchain-community langchain-openai pypdf

.env：
    DASHSCOPE_API_KEY=sk-xxx
    OPENAI_API_KEY=sk-xxx
"""

import os
import jieba
from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 固定动作：加载 .env
load_dotenv()

PDF_PATH = "/Users/lengchen/Downloads/恒生电子2026半年报.pdf"
RRF_K = 60          # RRF 常数，标准经验值
TOP_N = 10          # 每路召回数
FINAL_K = 3         # 最终返回数


# ============================================================
# 第一步：准备文档（真实PDF优先，没有就用模拟数据）
# ============================================================
def load_documents():
    if os.path.exists(PDF_PATH):
        print(f"📄 加载PDF：{PDF_PATH}")
        pages = PyPDFLoader(PDF_PATH).load()
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=300, chunk_overlap=50
        ).split_documents(pages)
    else:
        print(f"⚠️ 没找到 {PDF_PATH}，用模拟文本（特意加入代码/专有名词/数字）")
        mock = [
            "恒生电子(600570)2025年净利润12.3亿元，同比增长18.5%。AI大模型业务收入占比从5%提升至15%。",
            "恒生电子发布LightGPT金融大模型，在研报解读、合规审查、智能客服三大场景商业化落地。",
            "同花顺(300033)2025年营收89亿元，AI增值服务贡献占比42%，iFinD终端付费率提升11个百分点。",
            "平安银行上线140个智能体，贷款审批时效从3天缩短至4小时，风险识别准确率提升34%。",
            "东方财富(300059)聚焦财富管理，基金代销规模行业领先，互联网券商业务持续增长。",
        ]
        chunks = [Document(page_content=t, metadata={"source": "模拟研报", "id": i})
                  for i, t in enumerate(mock)]

    # 给每个 chunk 一个稳定 id（融合时用来对齐两路结果）
    for i, c in enumerate(chunks):
        c.metadata["doc_id"] = i
    print(f"✂️ 共 {len(chunks)} 个文本块\n")
    return chunks


chunks = load_documents()
texts = [c.page_content for c in chunks]


# ============================================================
# 第二步：BM25 关键词检索
# ============================================================
# 中文必须先分词！BM25 按"词"匹配，英文按空格，中文得用 jieba
# Java 类比：写入倒排索引前先 Tokenizer（Lucene 的 StandardTokenizer 同理）
tokenized_corpus = [list(jieba.cut(t)) for t in texts]
bm25 = BM25Okapi(tokenized_corpus)


def bm25_search(query: str, top_n: int = TOP_N):
    """返回 [(doc_id, rank), ...]，rank 从 1 开始"""
    tokens = list(jieba.cut(query))
    scores = bm25.get_scores(tokens)
    # 按分数降序，取 TopN（过滤掉0分的）
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ranked = [(doc_id, rank + 1) for rank, (doc_id, score)
              in enumerate(ranked[:top_n]) if score > 0]
    return ranked


# ============================================================
# 第三步：向量语义检索
# ============================================================
embeddings = DashScopeEmbeddings(
    model="text-embedding-v3",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)
print("向量化中...")
vector_db = FAISS.from_documents(chunks, embeddings)


def vector_search(query: str, top_n: int = TOP_N):
    """返回 [(doc_id, rank), ...]"""
    docs = vector_db.similarity_search_with_score(query, k=top_n)
    return [(doc.metadata["doc_id"], rank + 1) for rank, (doc, _) in enumerate(docs)]


# ============================================================
# 第四步：RRF 融合（只看排名，不看分数）
# ============================================================
def rrf_fusion(*rank_lists, k: int = RRF_K):
    """
    RRF 公式：score(doc) = Σ 1 / (k + rank)
    某文档在多路结果里都靠前 → 融合分高
    """
    fused_scores = {}
    for rank_list in rank_lists:
        for doc_id, rank in rank_list:
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank)
    # 按融合分降序
    return sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)


# ============================================================
# 第五步：Rerank 重排序（DashScope gte-rerank，失败则优雅降级）
# ============================================================
def rerank(query: str, candidate_ids):
    """用通义 gte-rerank 对候选逐对精算相关性，返回重排后的 doc_id 列表"""
    try:
        import dashscope
        from dashscope import TextReRank
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

        resp = TextReRank.call(
            model="gte-rerank",
            query=query,
            documents=[texts[i] for i in candidate_ids],
            top_n=len(candidate_ids),
            return_documents=False,
        )
        if resp.status_code != 200:
            raise Exception(resp.message)

        # output.results 里的 index 对应 candidates 的下标
        new_order = [candidate_ids[r["index"]] for r in resp.output["results"]]
        return new_order, True
    except Exception as e:
        print(f"  ⚠️ Rerank 不可用（{str(e)[:60]}），使用RRF原始顺序")
        return candidate_ids, False


# ============================================================
# 第六步：对比四种检索，肉眼看差异
# ============================================================
def show(query: str):
    print(f"\n{'='*70}")
    print(f"❓ {query}")
    print("="*70)

    bm_ids = bm25_search(query)
    vec_ids = vector_search(query)
    fused = rrf_fusion(bm_ids, vec_ids)
    candidate_ids = [doc_id for doc_id, _ in fused]

    reranked_ids, rerank_ok = rerank(query, candidate_ids)
    final_ids = reranked_ids[:FINAL_K]

    print(f"\n  BM25命中(关键词): {[texts[i][:18] for i,_ in bm_ids][:3]}")
    print(f"  向量命中(语义):   {[texts[i][:18] for i,_ in vec_ids][:3]}")
    print(f"  RRF融合Top:       {[texts[i][:18] for i in candidate_ids][:3]}")
    if rerank_ok:
        print(f"  Rerank后最终Top:  {[texts[i][:18] for i in final_ids]}")

    print("\n  📌 最终送入LLM的资料：")
    for rank, doc_id in enumerate(final_ids, 1):
        print(f"    {rank}. {texts[doc_id]}")


if __name__ == "__main__":
    # 1. 代码精确查询：BM25 应该明显强于向量
    show("600570 净利润多少")
    # 2. 专有名词：BM25 擅长
    show("LightGPT 用在什么场景")
    # 3. 数字精确匹配
    show("哪个业务同比增长18.5%")
    # 4. 语义类：向量更擅长，融合也兜得住
    show("恒生电子靠什么赚钱")
