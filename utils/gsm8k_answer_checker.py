import re
# for a given answer and a given ground truth, check if the answer is correct

# gsm8k uses #### to delimit answers
# it might be that theren is no exact match in which case we note this in the return
# returns

class gsm8k_answer_checker:
    async def check_answer(self, answer, ground_truth):
        # first remove think tags that might confuse the exact matching
        answer = self._remove_think_tags(answer)

        extracted_answer = self._extract_answer(answer)
        extracted_ground_truth = self._extract_answer(ground_truth)

        if not extracted_answer is None and not extracted_ground_truth is None:
            # if the extracted answer is the same as the extracted ground truth
            if extracted_answer - extracted_ground_truth < 1e-6:
                return {
                        "correct": True,
                         "mode": "match"
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
    def _remove_think_tags(self, answer):
        return re.sub(r'<think>.*?</think>', '', answer)

    def _extract_answer(self, text):
        """
        Extract the answer from the given text.

        Priority:
        1. Content inside \boxed{}
        2. Content after #### (ground truth)
        3. Last number in the text
        """
        # Match content inside \boxed{}
        boxed_match = re.search(r'\\boxed{(.*?)}', text)
        if boxed_match:
            return boxed_match.group(1)  # Return content inside braces
        
        # Match content after ####, assumes answer ends afterwards
        hash_match = re.search(r'####\s*(.*)', text)
        if hash_match:
            return hash_match.group(1).strip()  # Return content after ####
        
        # Match the last number in the text
        number_match = re.findall(r'\d+', text)
        if number_match:
            return number_match[-1]  # Return the last number matched
        
        return None  # Return None if no matches found


if __name__ == "__main__":
    checker = gsm8k_answer_checker()
    print(checker._extract_answer(test))
