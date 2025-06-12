import regex as re
import json
#from concurrent.futures import ThreadPoolExecutor
#from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool
from more_itertools import chunked

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
        # this one is easy w/o regex but it should rarely need to be used
        self.first_number_pattern = re.compile("\\d+")



    def check_batched(self, predictions: list[str], ground_truths: list[str]) -> list[dict]:
        if len(predictions) != len(ground_truths):
            raise Exception("Must have same quantity of predictions and ground truths")

        rewards = []
        #args = list(zip(predictions, ground_truths))
        batches = list(chunked(zip(predictions, ground_truths), 1000))

        '''
        #sequential
        for i, prediction in enumerate(predictions):
            rewards.append( self.check(prediction, ground_truths[i]) )
        '''
        '''
        # threads
        with ThreadPoolExecutor() as executor:
            rewards = list(executor.map(check, args))
        '''
        # processes
        with ProcessPoolExecutor() as executor:
            rewards = list(executor.map(check, args))

        with open("rewards_log.json", "w") as f:
            json.dump(rewards, f, indent=4)

        return rewards

#single prediction & ground truth 
def check(ac: AnswerChecker, prediction: str, ground_truth: str) -> dict:
    info = {
        "format_correct" : 0,
        "answer_correct" : 0,
        "reward" : 0
    }

    response_match = ac.response_pattern.search(prediction)
    ans = "x" # cannot match truth
    if response_match != None:
        info["format_correct"] = 1
        ans = response_match[4] 
    else:
        match = ac.first_number_pattern.search(prediction)
        if match == None:
            # TODO log parse fail
            return info # 0 reward for no match
        ans = match[0]

    truth_match = ac.truth_pattern.search(ground_truth)
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
    rewards = ac.check_batched(predictions, truths)
    #rewards = [check(ac, p, gt) for p, gt in zip(predictions, truths)]

profiler = cProfile.Profile()
profiler.enable()

test()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
