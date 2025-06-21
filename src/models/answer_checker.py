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
#helps run multiple tasks at once 

import cProfile #to measure how fast the code runs 
import pstats 
import logging #print messages to screen and files 


class AnswerChecker:
    def __init__(self):
        #set up basic logging once when object is created 
        logging.basicConfig(level=logging.INFO) #showing INFO level or more important levels
        #get logger for the file (used chat for this, maybe we wanna double check)
        self.logger = logging.getLogger(__name__)
        #__name__ contains name of current file 

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

        print(f"About to save {len(rewards)} rewards to file")
        with open("rewards_log.json", "w") as f:
            json.dump(rewards, f, indent=4)

        return rewards

    #single prediction & ground truth 
    def check(self, prediction: str, ground_truth: str) -> dict:
        print(f"Checking prediction: {prediction[:50]}...")
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
                self.logger.warning("No number found in prediction")
                return info # 0 reward for no match
            ans = match[0]

        truth_match = self.truth_format(ground_truth)
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


    def response_format(self, prediction: str):
        #match[0] = whole string, match[2] = inside think tags, match[4] = final answer
        match = re.search(r"(.*)<think>(.*?)</think>(.*?)\\boxed\{(.*)\}", prediction, re.DOTALL)
        return match
    
    def truth_format(self, ground_truth: str):
        #match[1] = final answer
        match = re.search("###\\s*(\\d+)", ground_truth)
        return match

    # TODO this will pull a number from inside the think tags, is that what we want?
    def first_number(self, prediction: str):
        #match[0] = first number in string
        match = re.search(r"\d+", prediction)
        return match


def test():

    with open("responses.txt", "r") as f:
        content = f.read()
    predictions = content.strip().split("\n\n")
    with open("ground_truths.txt", "r") as f:
        content = f.read()
    truths = content.strip().split("\n\n")

    ac = AnswerChecker()
    rewards = ac.check_batched(predictions, truths)

profiler = cProfile.Profile()
profiler.enable()

test()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(20)