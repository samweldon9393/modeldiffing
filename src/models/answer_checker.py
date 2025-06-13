import regex as re
import json

import cProfile
import pstats

from multiprocessing import Pool


#match[0] = whole string, match[2] = final answer
#response_pattern = re.compile(r"<think>(.*?)</think>\s*\\boxed{(.*)}")
response_pattern = re.compile(r"<think>(.*?)</think>\s*\\\\boxed{(.*)}")
#match[1] = final answer
truth_pattern = re.compile("###\\s*(\\d+)")
#match[0] = first number in string
# TODO this will pull a number from inside the think tags, is that what we want?
first_number_pattern = re.compile("\\d+")

def check_batch(batch):
    return [check(pred, gt) for pred, gt in batch]

def check_batched(predictions: list[str], ground_truths: list[str]) -> list[dict]:
    if len(predictions) != len(ground_truths):
        raise Exception("Must have same quantity of predictions and ground truths")

    #sequential
    rewards = [check(p, gt) for p, gt in zip(predictions, ground_truths)]

    with open("rewards_log.json", "w") as f:
        json.dump(rewards, f, indent=4)

    return rewards

#single prediction & ground truth 
def check(prediction: str, ground_truth: str) -> dict:
    info = {
        "format_correct" : 0,
        "answer_correct" : 0,
        "reward" : 0
    }

    print(repr(prediction))
    response_match = response_pattern.search(prediction)
    ans = "x" # cannot match truth
    if response_match != None:
        info["format_correct"] = 1
        ans = response_match[2] 
    else:
        match = first_number_pattern.search(prediction)
        if match == None:
            # TODO log parse fail
            return info # 0 reward for no match
        ans = match[0]
        # TODO first number probably gets the wrong answer more than last number would

    truth_match = truth_pattern.search(ground_truth)
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

    with open("short_responses.txt", "r") as f:
        content = f.read()
    predictions = content.strip().split("\n\n")
    with open("short_truths.txt", "r") as f:
        content = f.read()
    truths = content.strip().split("\n\n")

    rewards = check_batched(predictions, truths)

profiler = cProfile.Profile()
profiler.enable()

test()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
