import regex as re
import json
=======
#CHANGES MADE: implemented logging and improved error handling 
#1. added logging import and basic logging setup in __init__
#2. replaced all TODO comments with logging calls:
#   - logger.warning() for parsing failures and conversion errors
#   - logger.error() for ground truth parsing failures 
#3. fixed regex pattern in response_format():
#   - added re.DOTALL flag to handle newlines in <think> tages (chat gpt helped w this)
#   - fixed blackslash escaping: \\boxed instead of \\\\boxed 
#   - escaped braces: \{ and \} for proper matching 
#   - made pattern non-greedy: (.*?) instead of (.*) 
#4. added debug print statements to track problems  
#5. fixed file location issue and confirm JSON saves correctl

import re
import json
from concurrent.futures import ThreadPoolExecutor
#from concurrent.futures import ProcessPoolExecutor

import cProfile #to measure how fast the code runs 
import pstats 
import logging #print messages to screen and files 

from threading import Thread

class AnswerChecker:
    def __init__(self):
        #set up basic logging once when object is created 
        logging.basicConfig(level=logging.INFO) #showing INFO level or more important levels
        #get logger for the file (used chat for this, maybe we wanna double check)
        self.logger = logging.getLogger(__name__)
        #__name__ contains name of current file 

        '''
        regex options to be tested 
        # TODO what will the string literals look like? Two options:
        #response_pattern = re.compile(r"<think>(.*?)</think>\s*\\boxed{(.*)}")
        #kate's version: 
        #match = re.search(r"(.*)<think>(.*?)</think>(.*?)\\boxed\{(.*)\}", prediction, re.DOTALL)
        '''

        #match[0] = whole string, match[2] = final answer
        self.response_pattern = re.compile(r"<think>(.*?)</think>.*?\\\\boxed{(.*)}", re.DOTALL)

        #match[1] = final answer
        self.truth_pattern = re.compile("###\\s*(\\d+)")
        #match[0] = first number in string
        self.first_number_pattern = re.compile("\\d+")
        # TODO this would pull a number from inside the think tags, is that what we want?

        self.last_rewards = None 
        self.thread = None

    # kick off async checking
    def check_batched(self, predictions: list[str], ground_truths: list[str]) -> None:
        self._thread = Thread(target=self._check_worker, args=(predictions, ground_truths))
        self._thread.start()

    # can await the thread to get the rewards returned
    def wait(self) -> list[dict] | None:
        if self._thread:
            self._thread.join()
        return self.last_rewards

    # sequentially check the answers, log and store them in the object
    # TODO what to actually do with rewards? Apply directly? Need to wait for return?
    def _check_worker(self, predictions: list[str], ground_truths: list[str]):
        if len(predictions) != len(ground_truths):
            raise Exception("Must have same quantity of predictions and ground truths")

        #sequential
        rewards = [self.check(p, gt) for p, gt in zip(predictions, ground_truths)]

        with open("rewards_log.json", "w") as f:
            json.dump(rewards, f, indent=4)

        self.last_rewards = rewards

    #single prediction & ground truth 
    def check(self, prediction: str, ground_truth: str) -> dict:
        print(f"Checking prediction: {prediction[:50]}...")
        info = {
            "format_correct" : 0,
            "answer_correct" : 0,
            "reward" : 0
        }

        response_match = self.response_pattern.search(prediction)
        ans = "x" # cannot match truth
        if response_match != None:
            info["format_correct"] = 1
            ans = response_match[2] 
        else:
            match = self.first_number_pattern.search(prediction)
            if match == None:
                self.logger.warning("No number found in prediction")
                return info # 0 reward for no match
            ans = match[0]
            # TODO first number probably gets the wrong answer more than last number would

        truth_match = self.truth_pattern.search(ground_truth)
        if truth_match == None:
            self.logger.error("Could not parse ground truth")
            return info # ground truth parse failure

        try:
            if int(ans) == int(truth_match[1]):
                info["answer_correct"] = 1
        except(ValueError, TypeError):
            self.logger.warning("Could not convert answer to integer")
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
    ac.check_batched(predictions, truths)
    rewards = ac.wait()
    rewards = ac.check_batched(predictions, truths)

profiler = cProfile.Profile()
profiler.enable()

test()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)
