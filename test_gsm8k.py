from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset

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

model_name = "Qwen/Qwen2.5-1.5B-Instruct"

# ========= 1. 加载分词器和模型 =========
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

instruction = "Alice has 20 quarters. She wants to exchange them for nickels and so she goes to the bank. After getting back from the bank, she discovers that 20% of the nickels are iron nickels worth $3 each. What is the total value of her money now?"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user",   "content": instruction}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False
)

model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# ========= 3. 调用 generate 进行文本生成 =========
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=1024
)

# 只保留新生成的 token
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

print("=" * 60)
print("Qwen2.5 测试")
print("=" * 60)
print("\n[Instruction]\n", instruction)
print("\n[Response (模型生成的 Alpaca Response 部分)]\n", content)
print("=" * 60)