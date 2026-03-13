# 准备数据集
from datasets import load_dataset

# dataset = load_dataset("tatsu-lab/alpaca", split="train") # 适用于 alpaca 数据集
# dataset = load_dataset("yahma/alpaca-cleaned", split="train") # 适用于 alpaca-cleaned 数据集
dataset = load_dataset("rookiezyp/term", split="train") # 建筑施工术语

# 加载模型
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen3-8B-Base"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2", # "flash_attention_2 不支持T4"
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 配置LoRA
from peft import get_peft_model, LoraConfig
config = LoraConfig(
    r=8, # LoRA rank
    lora_alpha=16, # 缩放系数，通常为 2*r
    # target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # Attention+MLP
    use_dora=True,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()

# 格式化训练数据
# 转换为 prompt-completion 格式，SFTTrainer可以自动处理，只计算completion部分loss，mask掉prompt部分
def format_prompt_completion(example):
    instruction = example["instruction"]
    input_text  = example["input"]
    output      = example["output"]

    # 构造 prompt（用于输入部分）
    if input_text:
        prompt = (
            f"### Instruction: {instruction}\n"
            f"### Input: {input_text}\n"
            f"### Response:"
        )
    else:
        prompt = (
            f"### Instruction: {instruction}\n"
            f"### Response:"
        )

    # completion 只保留 output 肯定要有前导空格或 EOS token 确保模型分词正确
    completion = " " + output  # 空格让 tokenizer 区分 prompt 和 output

    return {"prompt": prompt, "completion": completion}
# 应用格式化，并删除原始字段
# formatted_dataset = dataset.map(format_prompt_completion, remove_columns=["instruction", "input", "output", "text"]) # 适用于 alpaca 数据集
formatted_dataset = dataset.map(format_prompt_completion, remove_columns=["instruction", "input", "output"]) # 适用于alpaca-cleaned数据集

# 配置SFTTrainer
from trl import SFTTrainer, SFTConfig

# SFT 配置
training_args = SFTConfig(
    output_dir="./qwen3-8b-base-dora-term",
    num_train_epochs=1,    # 指定训练轮数
    per_device_train_batch_size=4, # 每个设备上的 batch_size
    gradient_accumulation_steps=4, # 梯度累积步数
    # 有效 batch size = per_device_train_batch_size * gradient_accumulation_steps * GPU数量
    learning_rate=2e-4,
    fp16=False,
    bf16=True,  # 用 bfloat16 提升稳定性      
    logging_steps=10,
    save_strategy="epoch",
    optim="adamw_torch",  # 适配 QLoRA 的优化器      
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=0.3,
    max_length=512,  # 可增大，但依赖模型
    report_to="tensorboard",   # 可选 "wandb"、"tensorboard" 或 "none"
    completion_only_loss=True, # 只计算 completion 部分的 loss，默认为True
    push_to_hub=True,
    hub_model_id="rookiezyp/Qwen3-8B-Base-dora-term-20260313",
)

# 创建 SFT trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    args=training_args, # 这里传入 SFTConfig
)

trainer.train()

trainer.save_model("./qwen3-8b-base-dora-term-20260313")

trainer.push_to_hub() # 上传到 Hugging Face