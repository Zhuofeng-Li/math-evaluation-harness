from datasets import load_dataset
import os

# 定义数据集名称和输出路径
dataset_name = "TIGER-Lab/MMLU-Pro"
output_dir = "./data/mmlu_pro"
output_file = os.path.join(output_dir, "test.jsonl")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 定义格式化函数
def format_question_with_options(example):
    question = example['question']
    options = example['options']
    cot = example["cot_content"]
    
    # 将选项列表格式化为 A. B. C. 的形式
    formatted_options = [f"{chr(65+i)}. {option}" for i, option in enumerate(options)]
    formatted_question = f"{question}\n" + "\n".join(formatted_options)
    
    # 直接使用数据集中的 'answer' 列，它已经是字母格式
    correct_answer = example['answer']
    
    return {
        "question": formatted_question,
        "cot": cot,
        "answer": correct_answer
    }

print("开始加载和处理 MMLU-Pro 数据集的 'test' 分割...")

try:
    # 直接加载数据集的 'test' 分割
    test_ds = load_dataset(dataset_name, split="validation")
    
    print(f"  - 成功加载 'test' 分割，样本数量: {len(test_ds)}")
    
    # 应用格式化函数
    # 使用 'options' 和 'answer' 列
    formatted_ds = test_ds.map(format_question_with_options, remove_columns=test_ds.column_names)
    
    # 保存为单个 JSONL 文件
    formatted_ds.to_json(output_file, lines=True)
    
    print(f"\n成功将 MMLU-Pro 数据保存到 {output_file}")
    print(f"总样本数量: {len(formatted_ds)}")
    
except Exception as e:
    print(f"加载数据集时出错: {e}")
    print("请确保你已正确登录 Hugging Face 并拥有数据集访问权限。")