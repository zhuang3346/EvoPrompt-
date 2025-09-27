from typing import List, Dict, Any
import json
from llm_client import get_llm_client
from config import TASK_TYPE, LABEL_SPACE

class PromptEvaluator:
    """评估提示词性能的类"""
    
    def __init__(self, dev_set_path: str):
        self.dev_set = self._load_dev_set(dev_set_path)
    
    def _load_dev_set(self, file_path: str) -> List[Dict[str, Any]]:
        """加载开发集数据"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            raise ValueError(f"开发集文件未找到: {file_path}")
    
    def _is_output_correct(self, model_output: str, target_label: str) -> bool:
        """判断模型输出是否正确（根据任务类型定制）"""
        if TASK_TYPE == "sentiment_analysis":
            # 对于情感分析，检查输出是否包含目标标签
            return target_label.lower() in model_output.lower()
        else:
            # 默认简单匹配
            return target_label.lower() in model_output.lower()
    
    def evaluate_prompt(self, prompt: str) -> float:
        """评估单个提示词在开发集上的表现"""
        llm_client = get_llm_client()
        correct = 0
        total = len(self.dev_set)
        
        for example in self.dev_set:
            # 构建完整的输入，要求只输出英文标签
            full_input = (
                f"{prompt}\n\n"
                f"输入: {example['input']}\n"
                f"请只输出 positive 或 negative，不要输出其他内容。\n输出:"
            )
            
            # 获取模型输出
            model_output = llm_client.generate(full_input, temperature=0.0)
            
            if model_output and self._is_output_correct(model_output, example['target']):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0.0
        print(f"评估提示词: '{prompt[:50]}...' 准确率: {accuracy:.3f}")
        return accuracy
    
    def evaluate_population(self, prompts: List[str]) -> List[float]:
        """评估整个种群中所有提示词的表现"""
        return [self.evaluate_prompt(prompt) for prompt in prompts]
