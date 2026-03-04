from transformers import AutoModelForCausalLM, AutoTokenizer
from langsmith import evaluate, Client
from dotenv import load_dotenv
import os

load_dotenv()
client = Client()

dataset = client.read_dataset(dataset_name="gsm8k-100")

# 模型推理
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def extract_solution(text: str) -> str:
    solution = text.split("<SOLUTION>")[-1]
    solution = solution.split("</SOLUTION>")[0]
    return solution.strip()

def extract_thinking(text: str) -> str:
    thinking = text.split("<THINK>")[-1]
    thinking = thinking.split("</THINK>")[0]
    return thinking.strip()

def Qwen2_5_inference(inputs: dict):
    text = tokenizer.apply_chat_template(
        inputs["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    # 只保留新生成的 token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return {"thinking": extract_thinking(content), "solution": extract_solution(content)}

# 模型评估
def correctness(inputs: dict, outputs: dict, reference_outputs: dict)->bool:
    return outputs["solution"] == reference_outputs["answer"]

def format_correctness(outputs: dict)->bool:
    return outputs["solution"].isdigit()

# 1.运行实验exp_base
exp_base = evaluate(
    Qwen2_5_inference,
    data = dataset,
    evaluators=[correctness, format_correctness],
    experiment_prefix="qwen_base"
)