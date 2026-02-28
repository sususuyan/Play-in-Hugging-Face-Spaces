from langsmith import Client
from dotenv import load_dotenv

load_dotenv()

client = Client()

from datasets import load_dataset
# 随机采样100条数据
dataset = load_dataset("tatsu-lab/alpaca", split="train")
eval_dataset = dataset.shuffle(seed=42).select(range(100))

# 格式转换 alpaca->langsmith
def format_prompt(inst, inp):
    if inp.strip():
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{inst}

### Input:
{inp}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{inst}

### Response:
"""

# 创建数据集并上传数据
dataset_name = "alpaca-100"

dataset = client.create_dataset(dataset_name=dataset_name, description="Alpaca dataset with 100 examples")

examples = []
for data in eval_dataset:
    examples.append(
        {
            "inputs": {"question": format_prompt(data["instruction"], data["input"])},
            "outputs": {"response": data["output"]}
        }
    )

client.create_examples(
    dataset_id = dataset.id,
    examples = examples
)