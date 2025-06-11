import re

class AnswerChecker:
    def __init__(self):
        pass

    #single prediction & ground truth for now 
    def check(self, prediction, ground_truth):
        reward = 0 

    def format(self, prediction):
        #match[0] = whole string, match[2] = inside think tags, match[4] = final answer
        match = re.search("(.*)<think>(.*?)</think>(.*?)\\\\boxed{(.*)}", prediction)

ac = AnswerChecker()
ac.format('<think>1 plus 1 might equal 2</think>\\boxed{2}')
