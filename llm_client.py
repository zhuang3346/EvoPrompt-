from abc import ABC, abstractmethod
from openai import OpenAI
from typing import Optional

class BaseLLMClient(ABC):
    """LLM客户端的抽象基类，定义统一的接口规范"""
    
    @abstractmethod
    def generate(self, instruction: str, temperature: float = 0.7) -> Optional[str]:
        """向LLM API发送请求并返回生成的文本"""
        pass

class DeepSeekClient(BaseLLMClient):
    """DeepSeek API的具体实现"""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def generate(self, instruction: str, temperature: float = 0.7) -> Optional[str]:
        """实现DeepSeek API调用"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个有帮助的助手。"},
                    {"role": "user", "content": instruction}
                ],
                temperature=temperature,
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"DeepSeek API请求失败: {e}")
            return None

# 全局LLM客户端实例
_llm_client_instance = None

def init_llm_client(client: BaseLLMClient):
    """初始化全局LLM客户端"""
    global _llm_client_instance
    _llm_client_instance = client

def get_llm_client() -> BaseLLMClient:
    """获取全局LLM客户端实例"""
    if _llm_client_instance is None:
        raise ValueError("LLM客户端未初始化，请先调用init_llm_client()")
    return _llm_client_instance
