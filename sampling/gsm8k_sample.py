# Inference Pipeline Implementation Plan

# Arguments to be provided to the inference pipeline:
# 1. split: Specifies which dataset split to run inference on (e.g., 'train', 'test', or 'validation').
# 2. percentage: A float between 0 and 1 representing the percentage of data from the specified split to use for inference.
# 3. model: Specifies the model to use for inference. Options include:
#    - 'qwen-1.5b' for Qwen 1.5B
#    - 'llama3-ib-instruct' for LLaMA 3 IB Instruct
# 4. n: An integer representing how many times to sample an answer from the model for each question.

# Outputs and Logging:
# 1. For each question processed, the pipeline will log question-answer pairs in JSON format.
#    - Each JSON object will contain the question, all generated answers, and a boolean indicating correctness.
# 2. The JSON files will be saved incrementally during inference to avoid data loss and allow monitoring progress.

# Statistics to be calculated and recorded:
# 1. Accuracy metrics:
#    - pass@1: Percentage of correct predictions in the first sampled answer.
#    - pass@n/2: Percentage of correct predictions within the first n/2 samples.
#    - match@8: Percentage of correct predictions within the first 8 samples.
# 2. Answer length metrics:
#    - Average length of correct answers.
#    - Average length of incorrect answers.
#    - Overall average length of all answers.
# 3. Separate accuracy metrics for both training and testing splits (when applicable).

import json
import asyncio
import torch
from sglang import Template, Messages, gen, set_default_backend
from sglang.backends.huggingface import HuggingFaceBackend
from asyncio import Semaphore
# Set up Hugging Face backend with a local GPU-supported model
set_default_backend(HuggingFaceBackend(model="Qwen/Qwen1.5-7B-Chat", device="cuda"))

# Load questions from JSON
def load_questions(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

# Define sglang template
template = Template("""
{{#system}}
You are a helpful assistant solving math problems. You solve problems step by step using the following format:
                    
1. Put your step-by-step solution inside <think> tags, explaining each step clearly.
2. Verify your final answer whenever possible.
3. Provide the final answer in a <boxed> tag in a simplified and clear format.

Example 1:
Lucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?
<think>
1. Subtract the apples Lucy gave away: 18 - 4 = 14
2. Double the remaining apples: 14 * 2 = 28
</think>
\\boxed{28}

Example 2:
What is the value of (3 + 5) * 2?
<think>
1. Add 3 and 5 to get 8.
2. Multiply the result by 2: 8 * 2 = 16
</think>
\\boxed{16}

{{/system}}

{{#user}}
{{question}}
{{/user}}

{{#assistant}}
{{gen 'answer'}}
{{/assistant}}
""")

async def run_single_inference(question, semaphore, num_samples=8):
    async with semaphore:
        messages = Messages()
        messages += template.render(question=question)
        results = await gen.aparallel([messages], n=num_samples, temperature=0.7, top_p=0.9)
        return [output['answer'] for output in results[0]]

async def run_batched_inference(questions, batch_size=16, num_samples=8):
    semaphore = Semaphore(batch_size)
    tasks = [run_single_inference(q['question'], semaphore, num_samples) for q in questions]
    results = await asyncio.gather(*tasks)
    return {i: result for i, result in enumerate(results)}

if __name__ == "__main__":
    questions = load_questions("path_to_questions.json")
    results = asyncio.run(run_batched_inference(questions, batch_size=16, num_samples=8))
    print(results)
