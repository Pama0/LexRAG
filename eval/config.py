"""评测侧配置：judge LLM / embedding / 路径。

评测侧用 ragas 原生 llm_factory（instructor 结构化输出，collections 指标必需），
沿用 legacy 的智谱 GLM，与被测系统（项目 DeepSeek）解耦。
"""
import os

from dotenv import load_dotenv

load_dotenv()

# ── 路径 ──────────────────────────────────────────────
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(EVAL_DIR, "dataset")
TESTSET_PATH = os.path.join(DATASET_DIR, "testset.jsonl")
TESTSET_DRAFT_PATH = os.path.join(DATASET_DIR, "testset.draft.jsonl")
RESULTS_DIR = os.path.join(EVAL_DIR, "results")

# ── 评测侧模型 ────────────────────────────────────────
EVAL_LLM_MODEL = "glm-4-flash"
EVAL_LLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
EVAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"


def make_eval_llm():
    """评测 judge LLM：llm_factory + 智谱 OpenAI 兼容端点。"""
    import openai
    from ragas.llms import llm_factory

    client = openai.AsyncOpenAI(
        base_url=EVAL_LLM_BASE_URL,
        api_key=os.getenv("ZHIPU_API_KEY"),
    )
    return llm_factory(EVAL_LLM_MODEL, client=client)


def make_eval_embeddings():
    """评测 embedding：ragas HuggingFaceEmbeddings。"""
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=EVAL_EMBED_MODEL)
