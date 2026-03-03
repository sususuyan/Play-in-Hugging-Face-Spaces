from datasets import load_dataset, Dataset
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from trl import GRPOTrainer, GRPOConfig
import re

# 加载与预处理数据集 GSM8K
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

def get_gsm8k_questions():
    data = load_dataset('openai/gsm8k', 'main')["train"] 
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) 
    return data 

# 奖励函数
def extract_xml_answer(text: str) -> str:
    answer = text.split("<SOLUTION>")[-1]
    answer = answer.split("</SOLUTION>")[0]
    return answer.strip()

# 奖励函数
# 答案正确性奖励：与标准答案相等奖励2分，否则0分
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

# 答案格式奖励：答案是数字奖励0.5分，否则0分
def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

# 严格检查特定格式，要求格式完整，顺序正确，不能有额外内容，从头到尾匹配
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """严格检查特定格式"""
    pattern = r"^<THINK>\n.*?\n</THINK>\n<SOLUTION>\n.*?\n</SOLUTION>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 宽松检查特定格式，允许额外内容，不要求顺序正确，不要求从头到尾匹配
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """宽松检查特定格式"""
    pattern = r"<THINK>.*?</THINK>\s*<SOLUTION>.*?</SOLUTION>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# 计算XML格式得分：<THINK>和<SOLUTION>的个数和位置
def count_xml(text) -> float:
    count = 0.0
    if text.count("<THINK>\n") == 1:
        count += 0.125
    if text.count("\n</THINK>\n") == 1:
        count += 0.125
    if text.count("\n<SOLUTION>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</SOLUTION>\n")[-1])*0.001
    if text.count("\n</SOLUTION>") == 1:
        count += 0.125
        count -= (len(text.split("\n</SOLUTION>")[-1]) - 1)*0.001
    return count

# 计算XML格式得分：<THINK>和<SOLUTION>的个数和位置
def xml_count_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

if __name__ == "__main__":
    dataset = get_gsm8k_questions()

    # 加载模型
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2", # "flash_attention_2 支持L4"
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" 

    # 配置LoRA
    config = LoraConfig(
        r=8, # LoRA rank
        lora_alpha=16, # 缩放系数，通常为 2*r
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # GRPO 配置
    '''
    from vllm import SamplingParams
    vllm_sampling_params = SamplingParams(
        min_p = 0.1,
        top_p = 1.0,
        top_k = -1,
        seed = 3407,
        stop = [tokenizer.eos_token],
        include_stop_str_in_output = True,
    )
    '''
    training_args = GRPOConfig(
        # vllm_sampling_params = vllm_sampling_params,
        temperature = 1.0, # 控制采样随机性：1.0为标准，>1.0为更随机，<1.0为更确定
        learning_rate = 5e-6,
        weight_decay = 0.001,
        warmup_ratio = 0.1,
        lr_scheduler_type = "linear",
        optim = "adamw_torch",
        logging_steps = 10,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4, # 可增至 4 更稳
        num_generations = 4, # 每个prompt生成的样本数量，用于计算 reward 和策略梯度
        max_completion_length = 512,
        num_train_epochs = 1, # 完整训练可设为 1
        report_to = "tensorboard", # 也可传 "wandb"
        output_dir = "./Qwen2.5-1.5B-Instruct-GSM8K-GRPO",
        push_to_hub = True,
        hub_model_id = "rookiezyp/Qwen2.5-1.5B-Instruct-GSM8K-GRPO-20260303",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs=[
            xml_count_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func
        ],
        args = training_args,
        train_dataset = dataset,
    )

    trainer.train()
    trainer.save_model("./Qwen2.5-1.5B-Instruct-GSM8K-GRPO-20260303")
    trainer.push_to_hub() # 上传到 Hugging Face