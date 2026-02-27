from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def Qwen3_inference():
    # 指定要加载的 Hugging Face Models 上的模型名称
    model_name = "Qwen/Qwen3-1.7B-Base"

    # ========= 1. 加载分词器和模型 =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # ========= 2. 构造 Alpaca 格式的输入 =========
    # Alpaca 数据格式：Instruction + Response，这里只给出 Instruction，由模型生成 Response

    def build_alpaca_prompt(instruction: str) -> str:
        """构造 Alpaca 格式的提示文本（只包含 Instruction，不包含 Response）。"""
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
        return template


    # 示例指令（与示例中的 Alpaca 数据一致）
    instruction = "Give three tips for staying healthy."

    # 得到 Alpaca 格式的完整提示（模型将续写 ### Response: 之后的内容）
    alpaca_prompt = build_alpaca_prompt(instruction)

    # 以「用户单轮消息」的形式传入，使用 chat 模板转成模型输入格式
    messages = [
        {"role": "user", "content": alpaca_prompt}
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
        max_new_tokens=32768
    )

    # 只保留新生成的 token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    # ========= 4. 解析思维内容和最终回答（与 test_model.py 一致）=========
    try:
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    # ========= 5. 打印结果 =========
    print("=" * 60)
    print("Qwen3 测试")
    print("=" * 60)
    print("\n[Instruction]\n", instruction)
    print("\n[Thinking content]\n", thinking_content)
    print("\n[Response (模型生成的 Alpaca Response 部分)]\n", content)
    print("=" * 60)

def Qwen2_5_inference():
    # 指定要加载的 Hugging Face Models 上的模型名称
    model_name = "Qwen/Qwen2.5-1.5B"

    # ========= 1. 加载分词器和模型 =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )

    # ========= 2. 构造 Alpaca 格式的输入 =========
    # Alpaca 数据格式：Instruction + Response，这里只给出 Instruction，由模型生成 Response

    def build_alpaca_prompt(instruction: str) -> str:
        """构造 Alpaca 格式的提示文本（只包含 Instruction，不包含 Response）。"""
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
        return template


    # 示例指令（与示例中的 Alpaca 数据一致）
    instruction = "Give three tips for staying healthy."

    # 得到 Alpaca 格式的完整提示（模型将续写 ### Response: 之后的内容）
    alpaca_prompt = build_alpaca_prompt(instruction)

    model_inputs = tokenizer([alpaca_prompt], return_tensors="pt").to(model.device)

    # ========= 3. 调用 generate 进行文本生成 =========
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
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

def Qwen2_5_LoRA_inference():
    # 指定要加载的 Hugging Face Models 上的模型名称
    model_name = "Qwen/Qwen2.5-1.5B"
    lora_name = "rookiezyp/Qwen2.5-1.5B-alpaca-20260226"

    # ========= 1. 加载分词器和模型 =========
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, lora_name)

    # ========= 2. 构造 Alpaca 格式的输入 =========
    # Alpaca 数据格式：Instruction + Response，这里只给出 Instruction，由模型生成 Response

    def build_alpaca_prompt(instruction: str) -> str:
        """构造 Alpaca 格式的提示文本（只包含 Instruction，不包含 Response）。"""
        template = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
        )
        return template


    # 示例指令（与示例中的 Alpaca 数据一致）
    instruction = "Give three tips for staying healthy."

    # 得到 Alpaca 格式的完整提示（模型将续写 ### Response: 之后的内容）
    alpaca_prompt = build_alpaca_prompt(instruction)

    model_inputs = tokenizer([alpaca_prompt], return_tensors="pt").to(model.device)

    # ========= 3. 调用 generate 进行文本生成 =========
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )

    # 只保留新生成的 token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print("=" * 60)
    print("Qwen2.5 LoRA 测试")
    print("=" * 60)
    print("\n[Instruction]\n", instruction)
    print("\n[Response (模型生成的 Alpaca Response 部分)]\n", content)
    print("=" * 60)

if __name__ == "__main__":
    Qwen2_5_inference()