import os
import sys
from pathlib import Path
import json

# 添加当前目录到Python路径
sys.path.append(str(Path(__file__).parent))

from evolutionary_algorithm import EvolutionaryOptimizer
from llm_client import DeepSeekClient, init_llm_client

def setup_environment():
    """设置运行环境"""
    # 创建必要的目录
    os.makedirs("./data", exist_ok=True)
    os.makedirs("./results", exist_ok=True)
    
    # 检查初始提示词文件是否存在
    initial_prompts_file = "./data/initial_prompts.txt"
    if not os.path.exists(initial_prompts_file):
        # 创建示例初始提示词文件
        with open(initial_prompts_file, 'w', encoding='utf-8') as f:
            f.write("请分析这段文本的情感是正面还是负面。\n")
            f.write("判断以下文本的情感倾向：正面还是负面？\n")
            f.write("这是正面还是负面的评论？\n")
            f.write("请对这段文本进行情感分类。\n")
        print(f"已创建示例初始提示词文件: {initial_prompts_file}")
    
    # 检查开发集文件是否存在
    dev_set_file = "./data/dev_set.jsonl"
    if not os.path.exists(dev_set_file):
        # 创建示例开发集文件
        with open(dev_set_file, 'w', encoding='utf-8') as f:
            examples = [
                {"input": "这个产品非常好用，质量很棒！", "target": "positive"},
                {"input": "非常糟糕的体验，不会再买了。", "target": "negative"},
                {"input": "服务态度很好，解决问题很快。", "target": "positive"},
                {"input": "价格太贵，性价比不高。", "target": "negative"},
                {"input": "包装精美，送货速度快。", "target": "positive"}
            ]
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        print(f"已创建示例开发集文件: {dev_set_file}")

def main():
    """主函数"""
    print("初始化LLM提示词进化优化系统...")
    
    # 设置环境
    setup_environment()
    
    # 初始化DeepSeek客户端（替换为您的实际API密钥）
    deepseek_api_key = "sk-7b4e2837e6f6475ca066fd7f12252bb3"  # 请替换为您的实际API密钥
    deepseek_base_url = "https://api.deepseek.com"
    deepseek_model = "deepseek-chat"

    deepseek_client = DeepSeekClient(deepseek_api_key, deepseek_base_url, deepseek_model)
    init_llm_client(deepseek_client)
    
    # 创建并运行优化器
    optimizer = EvolutionaryOptimizer()
    
    print("开始进化优化过程...")
    best_prompt, best_score = optimizer.run_optimization()
    
    print("\n优化完成！")
    print(f"最终最佳提示词: {best_prompt}")
    print(f"最终最佳分数: {best_score:.3f}")
    
    # 保存结果
    results_dir = "./results"
    with open(f"{results_dir}/best_prompt.txt", 'w', encoding='utf-8') as f:
        f.write(f"最佳提示词: {best_prompt}\n")
        f.write(f"最佳分数: {best_score:.3f}\n")
    
    print(f"结果已保存到 {results_dir}/ 目录")

if __name__ == "__main__":
    main()
