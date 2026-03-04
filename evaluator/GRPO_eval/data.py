from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

from datasets import load_dataset
# 随机采样100条数据
dataset = load_dataset('openai/gsm8k', 'main')["train"] 
eval_dataset = dataset.shuffle(seed=42).select(range(100))

reasoning_start = "<THINK>"
reasoning_end   = "</THINK>"
solution_start  = "<SOLUTION>"
solution_end    = "</SOLUTION>"

SYSTEM_PROMPT = \
f"""You are given a math problem.
Think about the problem and provide your thinking process.
Place it between {reasoning_start} and {reasoning_end}.
Then provide the final answer as a single number.
Do NOT include any words, units, symbols, or explanations.
Place ONLY the number between {solution_start} and {solution_end}.
"""

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# 创建数据集并上传数据
dataset_name = "gsm8k-100"

dataset = client.create_dataset(dataset_name=dataset_name, description="GSM8K dataset with 100 examples")

examples = []
for data in eval_dataset:
    examples.append(
        {
            "inputs": 
            {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': data['question']}
                ]
            },
            "outputs": {"answer": extract_hash_answer(data["answer"])}
        }
    )

client.create_examples(
    dataset_id = dataset.id,
    examples = examples
)