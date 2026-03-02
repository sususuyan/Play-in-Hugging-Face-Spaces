from langsmith import evaluate, Client
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
client = Client()

# See the prompt: https://smith.langchain.com/hub/langchain-ai/pairwise-evaluation-2
prompt = client.pull_prompt("langchain-ai/pairwise-evaluation-2")
'''
原始 langchain-ai/pairwise-evaluation-2:
input_variables=['answer_a', 'answer_b', 'question'] input_types={} partial_variables={} metadata={'lc_hub_owner': 'langchain-ai', 'lc_hub_repo': 'pairwise-evaluation-2', 'lc_hub_commit_hash': '26647e49f48663aafd28478af7a4223dfe81ae2af7239fb3bb65b44122740517'} 
messages=[SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=[], input_types={}, partial_variables={}, template="Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. "), additional_kwargs={}), HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['answer_a', 'answer_b', 'question'], input_types={}, partial_variables={}, template="[User Question] {question}\n[The Start of Assistant A's Answer] {answer_a} [The End of Assistant A's Answer]\nThe Start of Assistant B's Answer] {answer_b} [The End of Assistant B's Answer]"), additional_kwargs={})] 
schema_={'type': 'object', 'title': 'Score', 'required': ['Preference'], 'properties': {'Preference': {'type': 'integer', 'description': 'Which assistant answer is preferred?'}}, 'description': 'After providing your explanation, output your final verdict by strictly following this format: \nOutput "1" if Assistant A answer is better based upon the factors above.\nOutput "2" if Assistant B answer is better based upon the factors above.\nOutput "0" if it is a tie.'} structured_output_kwargs={}
'''
# 增加"reference_answer"作为参考答案
prompt.input_variables.append("reference_answer")
new_system_prompt = """
Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Use the reference answer as guidance for intent and correctness, not as an exact template. Prefer responses that capture the core meaning and usefulness of the reference answer, even if phrasing differs.Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. 
"""
prompt.messages[0].prompt.template = new_system_prompt
prompt.messages[1].prompt.input_variables.append("reference_answer")
new_human_prompt = """
[User Question] {question}
[The Start of Assistant A's Answer] {answer_a} [The End of Assistant A's Answer]
[The Start of Assistant B's Answer] {answer_b} [The End of Assistant B's Answer]
[Reference Answer] {reference_answer}
"""
prompt.messages[1].prompt.template = new_human_prompt

# 评估模型
model = ChatOpenAI(model="Qwen/Qwen3-235B-A22B-Instruct-2507", temperature=0, api_key=os.getenv("SILICONFLOW_API_KEY"), base_url=os.getenv("SILICONFLOW_BASE_URL"))
chain = prompt | model

# 评估器
def ranked_preference(inputs: dict, outputs: list[dict], refernce_outputs: dict) -> list:
    # Assumes example inputs have a 'question' key and experiment
    # outputs have an 'answer' key.
    # reference_outputs have a 'response' key.
    response = chain.invoke({
        "question": inputs["question"],
        "answer_a": outputs[0].get("answer", "N/A"),
        "answer_b": outputs[1].get("answer", "N/A"),
        "reference_answer": reference_outputs["response"]
    })
    if response["Preference"] == 1:
        scores = [1, 0]
    elif response["Preference"] == 2:
        scores = [0, 1]
    else:
        scores = [0, 0]
    return scores

# 3.执行成对评估
evaluate(
    (exp_base, exp_lora),  
    evaluators=[ranked_preference],
    randomize_order=True,
    max_concurrency=4,
)

