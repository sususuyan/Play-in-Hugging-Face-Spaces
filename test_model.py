from transformers import AutoModelForCausalLM, AutoTokenizer

# 指定要加载的 Hugging Face Models 上的模型名称
model_name = "Qwen/Qwen3-1.7B"

# ========= 1. 加载分词器和模型 =========
# AutoTokenizer / AutoModelForCausalLM 会根据 model_name 自动选择合适的分词器和模型结构
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype="auto" 让框架自动选择合适的权重量化精度（比如 float16 / bfloat16），
    # 一般可以在保证精度的前提下降低显存占用。
    torch_dtype="auto",
    # device_map="auto" 会自动把模型加载到可用的设备上（如单/多块 GPU、CPU），
    # 对简单脚本来说可以避免手动指定 cuda 设备。
    device_map="auto"
)

# ========= 2. 构造对话形式的输入 =========
# 原始用户问题（文本形式），这里是英文的一个示例提示词
prompt = "Give me a short introduction to large language model."

# Qwen 的 chat 模型通常以消息列表（role + content）作为输入，
# role 一般包含 "system" / "user" / "assistant" 等。
messages = [
    {"role": "user", "content": prompt}
]

# 使用 tokenizer 自带的 chat 模板，把 messages 转换成模型实际接收的纯文本格式。
# - tokenize=False：只生成字符串，不直接返回 token id；
# - add_generation_prompt=True：在末尾添加模型生成用的提示（例如 <|assistant|>）；
# - enable_thinking=True：开启“思维模式”，让模型先生成隐藏的思维过程，再生成最终回答。
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # 在 Qwen3 中，用于切换是否让模型输出隐藏的思维链内容。
)

# 将上一步得到的文本再送入 tokenizer，转为模型需要的张量输入（input_ids 等），
# 并通过 .to(model.device) 把这些张量移动到和模型相同的设备上（比如 GPU）。
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# ========= 3. 调用 generate 进行文本生成 =========
# model.generate 会根据输入张量进行自回归生成。
# - max_new_tokens 限制本次生成的“新增 token”数量上限。
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)

# generated_ids 包含了「输入 + 模型生成」的完整序列。
# 这里通过切片的方式，只保留“新生成”的部分，丢弃原始输入 token：
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

# ========= 4. 从生成结果中解析思维内容和最终回答 =========
# 在 thinking 模式下，Qwen 会先生成一段“思维链”内容（被特殊标记包裹），
# 然后再生成用户可见的最终回答。
# 下面通过查找特殊 token id（151668，对应 </think>）来切分两部分内容。
try:
    # 从后往前查找 151668 的位置，相当于找到最后一个 </think> 的索引。
    # output_ids[::-1].index(151668) 是在反转列表后第一次出现 151668 的位置，
    # 然后用总长度减去这个偏移量，得到在原列表中的真实下标。
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    # 如果没有找到 151668（即模型没有输出思维链标记），
    # 就把 index 置为 0，表示后续全部都当作最终回答处理。
    index = 0

# 将 index 之前的 token 解码为“thinking content”，即模型的中间思考过程；
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")

# 将 index 之后的 token 解码为真正返回给用户看的“content” 内容。
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

# ========= 5. 打印结果 =========
# 为了便于观察，这里分别打印模型的“思维过程”和“最终回答”。
print("thinking content:", thinking_content)
print("content:", content)