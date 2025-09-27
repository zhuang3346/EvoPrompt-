from typing import List, Tuple
import random
from llm_client import get_llm_client
from config import (
    CROSSOVER_PROBABILITY, 
    MUTATION_PROBABILITY,
    INSTRUCTION_CROSSOVER,
    INSTRUCTION_MUTATION
)

class PromptPopulation:
    """管理提示词种群及其进化操作"""
    
    def __init__(self, prompts: List[str], scores: List[float]):
        if len(prompts) != len(scores):
            raise ValueError("提示词和分数列表长度必须一致")
        self.prompts = prompts
        self.scores = scores
    
    @classmethod
    def initialize_from_file(cls, file_path: str) -> 'PromptPopulation':
        """从文件初始化种群"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                prompts = [line.strip() for line in f if line.strip()]
            return cls(prompts, [0.0] * len(prompts))
        except FileNotFoundError:
            raise ValueError(f"初始提示词文件未找到: {file_path}")
    
    def get_best_prompt(self) -> Tuple[str, float]:
        """获取当前最佳提示词及其分数"""
        best_idx = self.scores.index(max(self.scores))
        return self.prompts[best_idx], self.scores[best_idx]
    
    def tournament_selection(self, tournament_size: int) -> str:
        """锦标赛选择：随机选择k个个体，返回得分最高的一个"""
        selected_indices = random.sample(range(len(self.prompts)), k=tournament_size)
        best_index = max(selected_indices, key=lambda i: self.scores[i])
        return self.prompts[best_index]
    
    def crossover(self, prompt1: str, prompt2: str) -> str:
        """使用LLM执行交叉操作"""
        llm_client = get_llm_client()
        instruction = INSTRUCTION_CROSSOVER.format(prompt1=prompt1, prompt2=prompt2)
        new_prompt = llm_client.generate(instruction, temperature=0.8)
        return new_prompt if new_prompt else random.choice([prompt1, prompt2])
    
    def mutate(self, prompt: str) -> str:
        """使用LLM执行变异操作"""
        llm_client = get_llm_client()
        instruction = INSTRUCTION_MUTATION.format(prompt=prompt)
        new_prompt = llm_client.generate(instruction, temperature=0.6)
        return new_prompt if new_prompt else prompt
    
    def create_new_generation(self, tournament_size: int, elitism_count: int = 1) -> List[str]:
        """创建新一代种群"""
        new_population = []
        
        # 精英保留：直接保留最佳个体
        elite_indices = sorted(range(len(self.scores)), key=lambda i: self.scores[i], reverse=True)[:elitism_count]
        new_population.extend([self.prompts[i] for i in elite_indices])
        
        # 生成剩余个体
        while len(new_population) < len(self.prompts):
            parent1 = self.tournament_selection(tournament_size)
            parent2 = self.tournament_selection(tournament_size)
            
            # 交叉
            if random.random() < CROSSOVER_PROBABILITY:
                child = self.crossover(parent1, parent2)
            else:
                child = random.choice([parent1, parent2])
            
            # 变异
            if random.random() < MUTATION_PROBABILITY:
                child = self.mutate(child)
            
            new_population.append(child)
        
        return new_population
    
    def update_scores(self, new_scores: List[float]):
        """更新种群中所有提示词的分数"""
        if len(new_scores) != len(self.prompts):
            raise ValueError("新分数列表长度与种群大小不一致")
        self.scores = new_scores
    
    def update_population(self, new_prompts: List[str], new_scores: List[float]):
        """完全更新种群"""
        if len(new_prompts) != len(new_scores):
            raise ValueError("新提示词和分数列表长度必须一致")
        self.prompts = new_prompts
        self.scores = new_scores
