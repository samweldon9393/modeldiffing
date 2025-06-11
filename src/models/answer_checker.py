import re

class AnswerChecker:
    def __init__(self):
        pass

    #single prediction & ground truth for now 
    def check(self, prediction, ground_truth):
        info = {
            "format_correct" : 0,
            "answer_correct" : 0,
            "reward" : 0
        }

        response_match = self.response_format(prediction)
        ans = "x" #cannot match truth
        if response_match != None:
            info["format_correct"] = 1
            ans = response_match[4] 
        else:
            ans = self.first_number(prediction)
            if ans == None:
                print("Failed to parse any number in prediction")
                return None

        truth_match = self.truth_format(ground_truth)
        if truth_match == None:
            print("Failed to parse truth")
            return None

        try:
            if int(ans) == int(truth_match[1]):
                info["answer_correct"] = 1
        except(ValueError, TypeError):
            print("Failed to parse")
            return None
            # ask Paul what we want to do here

        info["reward"] = (info["format_correct"] + info["answer_correct"]) / 2
        return info


    def response_format(self, prediction):
        #match[0] = whole string, match[2] = inside think tags, match[4] = final answer
        match = re.search("(.*)<think>(.*?)</think>(.*?)\\\\boxed{(.*)}", prediction)
        return match
    
    def truth_format(self, ground_truth):
        #match[1] = final answer
        match = re.search("###\\s*(\\d+)", ground_truth)
        return match

    def first_number(self, prediction):
        #match[0] = first number in string
        match = re.search(r"\d+", prediction)
        return match

ac = AnswerChecker()

#response = "<think>1 plus 1 might equal 2</think> \\boxed{2} adasd"
response = "hello 1 friend  sasdfaf"
ground_truth = " ... ###  asdfasdf"

info = ac.check(response, ground_truth)

print(info)
