from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langsmith import evaluate, Client
from dotenv import load_dotenv
import os

load_dotenv()
client = Client()

dataset = client.read_dataset(dataset_name="alpaca-100")

# 待评估函数
model_name = "Qwen/Qwen2.5-1.5B"
lora_name = "rookiezyp/Qwen2.5-1.5B-alpaca-20260226"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_name)

def Qwen2_5_LoRA_inference(inputs: dict):
    model_inputs = tokenizer([inputs["question"]], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    # 只保留新生成的 token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return {"answer": content}

# 2.运行实验exp_lora
exp_lora = evaluate(
    Qwen2_5_LoRA_inference,
    data = dataset,
    experiment_prefix="qwen_lora"
)