from transformers import AutoTokenizer, AutoModelForCausalLM

# 指定要加载的 Hugging Face Models 上的模型名称
model_name = "Qwen/Qwen2.5-1.5B"

# ========= 1. 加载分词器和模型 =========
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

repo_id = "rookiezyp/Qwen2.5-1.5B-ori"
model.push_to_hub(repo_id)
tokenizer.push_to_hub(repo_id)

print(f"Uploaded to https://huggingface.co/{repo_id}")