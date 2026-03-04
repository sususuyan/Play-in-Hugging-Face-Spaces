from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from peft import PeftModel

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

def Qwen2_5_inference():
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # ========= 1. 加载分词器和模型 =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    instruction = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

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

def Qwen2_5_GRPO_inference():
    # 指定要加载的 Hugging Face Models 上的模型名称
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    lora_name = "rookiezyp/Qwen2.5-1.5B-Instruct-GSM8K-GRPO-20260303"

    # ========= 1. 加载分词器和模型 =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_name)

    instruction = "James writes a 3-page letter to 2 different friends twice a week. How many pages does he write a year?"

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
    print("Qwen2.5 GRPO 测试")
    print("=" * 60)
    print("\n[Instruction]\n", instruction)
    print("\n[Response (模型生成的 Alpaca Response 部分)]\n", content)
    print("=" * 60)

if __name__ == "__main__":
    Qwen2_5_inference()
    Qwen2_5_GRPO_inference()