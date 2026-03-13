from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from langsmith import evaluate, Client
from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI

load_dotenv()

# 待评估函数
model_name = "Qwen/Qwen3-8B-Base"
lora_name = "rookiezyp/Qwen3-8B-Base-dora-term-20260313"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, lora_name)

def Qwen3_LoRA_inference(inputs: dict):
    model_inputs = tokenizer([inputs["question"]], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )

    # 只保留新生成的 token
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    return {"answer": content}

# LLM-as-judge instructions
grader_instructions = """You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting statements.
(3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the  ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct."""

# LLM-as-judge output schema
class Grade(TypedDict):
    """Compare the expected and actual answers and grade the actual answer."""
    reasoning: Annotated[str, ..., "Explain your reasoning for whether the actual response is correct or not."]
    is_correct: Annotated[bool, ..., "True if the student response is mostly or exactly correct, otherwise False."]

# Judge LLM
grader_llm = ChatOpenAI(
    model="Pro/MiniMaxAI/MiniMax-M2.5", 
    temperature=0, 
    max_tokens=1024,
    api_key=os.getenv("SILICONFLOW_API_KEY"), 
    base_url=os.getenv("SILICONFLOW_BASE_URL")
)
grader_llm = grader_llm.with_structured_output(Grade, method="json_schema", strict=True)

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    """Evaluate if the final response is equivalent to reference response."""

    # Note that we assume the outputs has a 'response' dictionary. We'll need to make sure
    # that the target function we define includes this key.
    user = f"""QUESTION: {inputs['question']}
    GROUND TRUTH RESPONSE: {reference_outputs['response']}
    STUDENT RESPONSE: {outputs['answer']}"""

    grade = grader_llm.invoke([{"role": "system", "content": grader_instructions}, {"role": "user", "content": user}])
    return grade["is_correct"]

def main():
    client = Client()
    experiment_results = client.evaluate(
        Qwen3_LoRA_inference,
        data="term-100",
        evaluators=[
            correctness_evaluator,
        ],
        experiment_prefix="Qwen3-8B-DoRA-correctness-eval"
    )

if __name__ == "__main__":
    main()