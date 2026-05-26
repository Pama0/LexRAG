import os

from dotenv import load_dotenv
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import Settings

load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')
zhipu_api_key = os.getenv('ZHIPU_API_KEY')
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
def configure_llm():
    """配置 LLM 并设置全局参数

    DeepSeek 的 thinking 模型（如 v4-flash）会返回 reasoning_content 字段，
    且要求多轮调用时回传——LlamaIndex 当前不支持该字段往返，会导致
    400 'reasoning_content must be passed back'。
    禁用 thinking 即可走标准 chat completions 流程。
    """
    llm = OpenAILike(
        model="deepseek-v4-flash",
        api_base="https://api.deepseek.com/v1",
        api_key=deepseek_api_key,
        context_window=128000,
        is_chat_model=True,
        is_function_calling_model=True,
        additional_kwargs={
            # 关闭 thinking 模式（DeepSeek 官方字段：thinking.type = enabled/disabled）
            # 默认 enabled；LlamaIndex 当前不支持 reasoning_content 字段往返，必须关闭
            "extra_body": {"thinking": {"type": "disabled"}},
        },
    )
    Settings.llm = llm
    return llm
