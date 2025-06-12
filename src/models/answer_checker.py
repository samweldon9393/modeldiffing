import re
import json
from concurrent.futures import ThreadPoolExecutor
#from concurrent.futures import ProcessPoolExecutor

import cProfile
import pstats


class AnswerChecker:
    def __init__(self):
        #match[0] = whole string, match[2] = inside think tags, match[4] = final answer
        self.response_pattern = re.compile("(.*)<think>(.*?)</think>(.*?)\\\\boxed{(.*)}")
        #match[1] = final answer
        self.truth_pattern = re.compile("###\\s*(\\d+)")
        #match[0] = first number in string
        # TODO this will pull a number from inside the think tags, is that what we want?
        self.first_number_pattern = re.compile("\\d+")


    def check_batched(self, predictions: list[str], ground_truths: list[str]) -> list[dict]:
        if len(predictions) != len(ground_truths):
            raise Exception("Must have same quantity of predictions and ground truths")

        rewards = []

        '''
        #sequential
        for i, prediction in enumerate(predictions):
            rewards.append( self.check(prediction, ground_truths[i]) )
        '''
        # threads
        with ThreadPoolExecutor() as executor:
            rewards = list(executor.map(self.check, predictions, ground_truths))
        '''
        # processes
        with ProcessPoolExecutor() as executor:
            rewards = list(executor.map(self.check, predictions, ground_truths))
        '''

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

        response_match = self.response_pattern.search(prediction)
        ans = "x" # cannot match truth
        if response_match != None:
            info["format_correct"] = 1
            ans = response_match[4] 
        else:
            match = self.first_number_pattern.search(prediction)
            if match == None:
                # TODO log parse fail
                return info # 0 reward for no match
            ans = match[0]

        truth_match = self.truth_pattern.search(ground_truth)
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


def test():

    with open("responses.txt", "r") as f:
        content = f.read()
    predictions = content.strip().split("\n\n")
    with open("ground_truths.txt", "r") as f:
        content = f.read()
    truths = content.strip().split("\n\n")

    ac = AnswerChecker()
    #rewards = ac.check_batched(predictions, truths)
    rewards = [ac.check(p, gt) for p, gt in zip(predictions, truths)]

profiler = cProfile.Profile()
profiler.enable()

test()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
