import re
import json

class AnswerChecker:
    def __init__(self):
        pass

    def check_batched(self, predictions: list[str], ground_truths: list[str]) -> list[dict]:
        if len(predictions) != len(ground_truths):
            raise Exception("Must have same quantity of predictions and ground truths")

        rewards = []
        for i, prediction in enumerate(predictions):
            rewards.append( self.check(prediction, ground_truths[i]) )

        with open("rewards_log.json", "w") as f:
            json.dump(rewards, f, indent=4)

        return rewards

    #single prediction & ground truth 
    def check(self, prediction: str, ground_truth: str) -> dict:
        info = {
            "format_correct" : 0,
            "answer_correct" : 0,
            "reward" : 0
        }

        response_match = self.response_format(prediction)
        ans = "x" # cannot match truth
        if response_match != None:
            info["format_correct"] = 1
            ans = response_match[4] 
        else:
            match = self.first_number(prediction)
            if match == None:
                # TODO log parse fail
                return info # 0 reward for no match
            ans = match[0]

        truth_match = self.truth_format(ground_truth)
        if truth_match == None:
            # TODO log parse fail
            return info # ground truth parse failure

        try:
            if int(ans) == int(truth_match[1]):
                info["answer_correct"] = 1
        except(ValueError, TypeError):
            # TODO log parse fail
            return info # integer parse error
            # ask Paul what we want to do here

        info["reward"] = (info["format_correct"] + info["answer_correct"]) / 2
        return info


    def response_format(self, prediction: str):
        #match[0] = whole string, match[2] = inside think tags, match[4] = final answer
        match = re.search("(.*)<think>(.*?)</think>(.*?)\\\\boxed{(.*)}", prediction)
        return match
    
    def truth_format(self, ground_truth: str):
        #match[1] = final answer
        match = re.search("###\\s*(\\d+)", ground_truth)
        return match

    def first_number(self, prediction: str):
        #match[0] = first number in string
        match = re.search(r"\d+", prediction)
        return match

ac = AnswerChecker()

responses = ["hello 2 friend  sasdfaf", "<think>1 plus 1 might equal 2</think> \\boxed{2} adasd"]
ground_truths = [" ... ### 2", "this is the answer ### 2"]

info = ac.check_batched(responses, ground_truths)


print(info)

