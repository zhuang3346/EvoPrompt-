from typing import Tuple, Optional
from population import PromptPopulation
from evaluator import PromptEvaluator
from config import (
    POPULATION_SIZE,
    MAX_ITERATIONS,
    TOURNAMENT_SIZE,
    ELITISM_COUNT,
    INITIAL_PROMPTS_FILE,
    DEV_SET_PATH
)

class EvolutionaryOptimizer:
    """进化算法优化引擎"""
    
    def __init__(self):
        self.population = PromptPopulation.initialize_from_file(INITIAL_PROMPTS_FILE)
        self.evaluator = PromptEvaluator(DEV_SET_PATH)
        self.best_prompt = None
        self.best_score = -1.0
        self.history = []
    
    def _update_best(self, current_best_score: float, current_best_prompt: str):
        """更新历史最佳记录"""
        if current_best_score > self.best_score:
            self.best_score = current_best_score
            self.best_prompt = current_best_prompt
            print(f"发现新的最佳提示词! 分数: {self.best_score:.3f}")
            print(f"最佳提示词: {self.best_prompt}")
    
    def run_optimization(self, max_iterations: int = MAX_ITERATIONS) -> Tuple[str, float]:
        """执行进化优化主循环"""
        
        # 初始评估
        print("开始初始种群评估...")
        initial_scores = self.evaluator.evaluate_population(self.population.prompts)
        self.population.update_scores(initial_scores)
        
        current_best_prompt, current_best_score = self.population.get_best_prompt()
        self._update_best(current_best_score, current_best_prompt)
        self.history.append((self.population.prompts.copy(), initial_scores.copy()))
        
        # 进化循环
        for iteration in range(max_iterations):
            print(f"\n开始第 {iteration + 1}/{max_iterations} 代进化...")
            
            # 创建新一代
            new_prompts = self.population.create_new_generation(TOURNAMENT_SIZE, ELITISM_COUNT)
            
            # 评估新一代
            new_scores = self.evaluator.evaluate_population(new_prompts)
            
            # 更新种群
            self.population.update_population(new_prompts, new_scores)
            
            # 更新历史最佳
            current_best_prompt, current_best_score = self.population.get_best_prompt()
            self._update_best(current_best_score, current_best_prompt)
            
            # 记录历史
            self.history.append((new_prompts.copy(), new_scores.copy()))
            
            print(f"第 {iteration + 1} 代完成，当前最佳分数: {self.best_score:.3f}")
        
        return self.best_prompt, self.best_score
    
    def get_optimization_history(self):
        """获取优化过程的历史记录"""
        return self.history
