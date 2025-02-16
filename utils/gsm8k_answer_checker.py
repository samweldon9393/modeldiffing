import re
import asyncio
# for a given answer and a given ground truth, check if the answer is correct

# gsm8k uses #### to delimit answers
# it might be that theren is no exact match in which case we note this in the return
# returns

class gsm8k_answer_checker:

    @staticmethod
    async def eval(outputs):
        n = len(outputs[0]["answers"])
        for output in outputs:
            for answer in output['answers']:
                answer['answer_eval'] = await gsm8k_answer_checker.check_answer(answer['text'], output['ground_truth'])
            
            acc = sum([answer['answer_eval']['correct'] for answer in output['answers']]) / n # fraction of answers that are correct
            output['evaluation'] = {
                "accuracy": acc,
                "pass@n": True if acc > 0 else False,
                "match@n": True if acc > 0.5 else False
            }
        return outputs

    @staticmethod
    async def check_answer(answer, ground_truth):
        # first remove think tags that might confuse the exact matching
        answer = gsm8k_answer_checker._remove_think_tags(answer)

        extracted_answer = gsm8k_answer_checker._extract_answer(answer)
        extracted_ground_truth = gsm8k_answer_checker._extract_answer(ground_truth)

        if not extracted_answer is None and not extracted_ground_truth is None:
            # if the extracted answer is the same as the extracted ground truth
            if extracted_answer - extracted_ground_truth < 1e-6:
                return {
                        "correct": True,
                         "mode": "match",
                         "extracted_answer": extracted_answer,
                            "extracted_ground_truth": extracted_ground_truth
                    }
            # if the extracted answer is not the same as the extracted ground truth
            else:
                return {
                        "correct": False,
                        "mode": "match",
                        "extracted_answer": extracted_answer,
                        "extracted_ground_truth": extracted_ground_truth
                    }
        else:
            return {
                    "correct": False,
                    "mode": "no_match",
                    "extracted_answer": extracted_answer,
                    "extracted_ground_truth": extracted_ground_truth
                } 


    # removes text between <think> tags
    @staticmethod
    def _remove_think_tags(answer):
        return re.sub(r'<think>.*?</think>', '', answer)

    @staticmethod
    def _extract_answer(text):
        """
        Extract the answer from the given text.

        Priority:
        1. Content inside \boxed{}
        2. Content after #### (ground truth)
        3. Last number in the text
        """
        def clean_number(num_str):
            # Remove any non-numeric characters except '.' and '-'
            cleaned = re.sub(r'[^\d.-]', '', num_str)
            try:
                return float(cleaned)
            except ValueError:
                return None

        # Match content inside \boxed{}
        boxed_match = re.search(r'\\boxed{([\d.,]+(?:\.\d+)?)\D*}', text)
        if boxed_match:
            return clean_number(boxed_match.group(1))
        
        # Match content after ####
        hash_match = re.search(r'####\s*(.*)', text)
        if hash_match:
            return clean_number(hash_match.group(1).strip())
        
        # Match all numbers, including decimals and commas
        number_matches = re.findall(r'[\d.,]+(?:\.\d+)?', text)
        if number_matches:
            # Take the last number, clean it, and convert to float
            return clean_number(number_matches[-1])

        return None  # Return None if no matches found


async def main():
    checker = gsm8k_answer_checker()
    test = [{'question': "\nYou are a helpful assistant solving math problems. Solve problems step by step using the following format:\n\n1. Put your step-by-step solution inside <think> tags, explaining each step clearly.\n2. Verify your final answer whenever possible.\n3. Provide the final answer in a \\boxed{} tag in a simplified and clear format.\n\nExample 1:\nLucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?\n<think>\n1. Subtract the apples Lucy gave away: 18 - 4 = 14\n2. Double the remaining apples: 14 * 2 = 28\n</think>\n\\boxed{28}\n<|endoftext|>\n\nExample 2:\nWhat is the value of (3 + 5) * 2?\n<think>\n1. Add 3 and 5 to get 8.\n2. Multiply the result by 2: 8 * 2 = 16\n</think>\n\\boxed{16}\n<|endoftext|>\n\nQuestion:\nJanet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\nAnswer:\n", 'ground_truth': 'Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer’s market.\n#### 18', 'answers': [{'text': '<think>\n1. Calculate the total number of eggs laid by the ducks: 16 eggs/day\n2. Calculate the total number of eggs eaten by Janet and baked for her friends: 3 (breakfast) + 4 (muffins) = 7 eggs\n3. Calculate the number of eggs left for sale: 16 (eggs laid) - 7 (eggs eaten/baked) = 9 eggs\n4. Multiply the number of eggs left by the price per egg: 9 eggs * $2/egg = $18\n</think>\n\\boxed{\\$18}\n', 'meta_info': {'id': '9c3b74039fae4c53afabddc1e8886ffc', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 294, 'completion_tokens': 132, 'cached_tokens': 293}}, {'text': "<think>\n1. Calculate the total number of eggs laid by the ducks: 16 eggs/day\n2. Calculate the total number of eggs eaten for breakfast: 3 eggs/day\n3. Calculate the total number of eggs used for baking muffins: 4 eggs/day\n4. Calculate the total number of eggs left for the farmers' market: 16 - 3 - 4 = 9 eggs/day\n5. Calculate the total daily income from selling the eggs at the farmers' market: 9 eggs/day * $2/egg = $18/day\n</think>\n\\boxed{\\$18}\n", 'meta_info': {'id': 'd6536c3003fd46e3bdde990403b986f3', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 294, 'completion_tokens': 133, 'cached_tokens': 293}}], 'evaluation': {'answer_1': {'text': '<think>\n1. Calculate the total number of eggs laid by the ducks: 16 eggs/day\n2. Calculate the total number of eggs eaten by Janet and baked for her friends: 3 (breakfast) + 4 (muffins) = 7 eggs\n3. Calculate the number of eggs left for sale: 16 (eggs laid) - 7 (eggs eaten/baked) = 9 eggs\n4. Multiply the number of eggs left by the price per egg: 9 eggs * $2/egg = $18\n</think>\n\\boxed{\\$18}\n', 'meta_info': {'id': '9c3b74039fae4c53afabddc1e8886ffc', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 294, 'completion_tokens': 132, 'cached_tokens': 293}}, 'answer_2': {'text': "<think>\n1. Calculate the total number of eggs laid by the ducks: 16 eggs/day\n2. Calculate the total number of eggs eaten for breakfast: 3 eggs/day\n3. Calculate the total number of eggs used for baking muffins: 4 eggs/day\n4. Calculate the total number of eggs left for the farmers' market: 16 - 3 - 4 = 9 eggs/day\n5. Calculate the total daily income from selling the eggs at the farmers' market: 9 eggs/day * $2/egg = $18/day\n</think>\n\\boxed{\\$18}\n", 'meta_info': {'id': 'd6536c3003fd46e3bdde990403b986f3', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 294, 'completion_tokens': 133, 'cached_tokens': 293}}}}, {'question': '\nYou are a helpful assistant solving math problems. Solve problems step by step using the following format:\n\n1. Put your step-by-step solution inside <think> tags, explaining each step clearly.\n2. Verify your final answer whenever possible.\n3. Provide the final answer in a \\boxed{} tag in a simplified and clear format.\n\nExample 1:\nLucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?\n<think>\n1. Subtract the apples Lucy gave away: 18 - 4 = 14\n2. Double the remaining apples: 14 * 2 = 28\n</think>\n\\boxed{28}\n<|endoftext|>\n\nExample 2:\nWhat is the value of (3 + 5) * 2?\n<think>\n1. Add 3 and 5 to get 8.\n2. Multiply the result by 2: 8 * 2 = 16\n</think>\n\\boxed{16}\n<|endoftext|>\n\nQuestion:\nA robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?\n\nAnswer:\n', 'ground_truth': 'It takes 2/2=<<2/2=1>>1 bolt of white fiber\nSo the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric\n#### 3', 'answers': [{'text': '<think>\n1. Calculate the amount of white fiber: half of 2 is 1.\n2. Add the amount of blue and white fiber: 2 + 1 = 3\n</think>\n\\boxed{3} ', 'meta_info': {'id': '4d15a037422a49a3a003c8818013324f', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 256, 'completion_tokens': 55, 'cached_tokens': 255}}, {'text': '<think>\n1. Determine the amount of white fiber needed: half of 2 bolts is 1 bolt.\n2. Add the blue and white fiber needed: 2 (blue) + 1 (white) = 3\n</think>\n\\boxed{3}\n', 'meta_info': {'id': '7833ac6f58b34ed8a6d0310710cfa5d5', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 256, 'completion_tokens': 63, 'cached_tokens': 255}}], 'evaluation': {'answer_1': {'text': '<think>\n1. Calculate the amount of white fiber: half of 2 is 1.\n2. Add the amount of blue and white fiber: 2 + 1 = 3\n</think>\n\\boxed{3} ', 'meta_info': {'id': '4d15a037422a49a3a003c8818013324f', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 256, 'completion_tokens': 55, 'cached_tokens': 255}}, 'answer_2': {'text': '<think>\n1. Determine the amount of white fiber needed: half of 2 bolts is 1 bolt.\n2. Add the blue and white fiber needed: 2 (blue) + 1 (white) = 3\n</think>\n\\boxed{3}\n', 'meta_info': {'id': '7833ac6f58b34ed8a6d0310710cfa5d5', 'finish_reason': {'type': 'stop', 'matched': '<|endoftext|>'}, 'prompt_tokens': 256, 'completion_tokens': 63, 'cached_tokens': 255}}}}]
    results = await checker.eval(test)
    for result in results:
        print(result["evaluation"])
        for answer in result["answers"]:
            print(answer["answer_eval"])
if __name__ == "__main__":
    asyncio.run(main())

