# 进化算法超参数
POPULATION_SIZE = 10                # 种群规模
MAX_ITERATIONS = 10                 # 最大迭代次数
TOURNAMENT_SIZE = 3                 # 锦标赛选择的个体数量
CROSSOVER_PROBABILITY = 0.9         # 交叉操作的概率
MUTATION_PROBABILITY = 0.1          # 变异操作的概率
ELITISM_COUNT = 1                   # 保留的最佳个体数量

# 任务相关配置
TASK_TYPE = "sentiment_analysis"
LABEL_SPACE = ["negative", "positive"]

# 初始提示词种群文件路径
INITIAL_PROMPTS_FILE = "./data/initial_prompts.txt"

# 开发集数据路径
DEV_SET_PATH = "./data/dev_set.jsonl"

# LLM指令模板
INSTRUCTION_CROSSOVER = """
你是一个专业的提示词优化专家。请将以下两个提示词的优点融合，创建一个全新的、连贯且有效的提示词。
提示词1: {prompt1}
提示词2: {prompt2}
请输出融合后的新提示词，不要包含任何其他内容。
"""

INSTRUCTION_MUTATION = """
你是一个专业的提示词优化专家。请对以下提示词进行小幅改进，使其更加有效、清晰和简洁。
原始提示词: {prompt}
请输出改进后的新提示词，不要包含任何其他内容。
"""
