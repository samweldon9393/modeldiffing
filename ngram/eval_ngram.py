import argparse
import json
import re
from tqdm import tqdm
from ngram.prompt_ngram import prompt_ngram, eval_predictions

# paul@paul-ThinkPad:~/Desktop/minireason$ python3 -m ngram.eval_ngram --path=test_evaluation.json --n=2

"""
args:
    path (str): Path to the json file with question answer pairs
    n (int): The n the ngram models should use
    lambdas (List[float]): The lambdas to use for interpolation
    train_on_pred (bool): Whether to train the ngram model while predicting
    train_on_prompt (bool): Whether to train the ngram model on the prompt
    use_tokenizer (bool): Whether to use a tokenizer to split the text
"""

def eval_json(path, n, lambdas, lookahead, beam_width, train_on_pred, train_on_prompt):
    
    n_tokens = [0, 0, 0, 0]
    answers_accuracy = [0, 0, 0, 0]

    with open(path, 'r') as f:
        data = json.load(f)
    for d in tqdm(data[:100]):
        
        # Initialize the n-gram model.
        model = prompt_ngram(n, lambdas)
        
        answers = d["answers"]

        if (train_on_prompt):
            # strip the prompt and only keep the question
            question = re.search(r'Question:\s*(.*?)\s*Answer:', d["question"], re.DOTALL)
            if question:
                question = question.group(1).strip()  # Remove extra spaces and newlines
            else:
                # can't match format take the whole question
                question = d["question"]
            
            model.train(question)

        for i, answer in enumerate(answers):
            answer = answer["text"]
            num_tokens = len(answer.split())
            n_tokens[i] += num_tokens

            if num_tokens != 0:
                pred_answer = model.predict(answer, train=train_on_pred, num_tokens=lookahead, beam_width=beam_width)
                eval_acc = eval_predictions(pred_answer, answer, num_tokens=lookahead)
                answers_accuracy[i] += eval_acc * num_tokens  # Weighted sum

                if not train_on_pred:
                    model.train(answer)

    
    accuracy = {
        f"answer_{i}": answers_accuracy[i] / n_tokens[i] if n_tokens[i] != 0 else 0
        for i in range(4)
    }


    print(accuracy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--n', type=int, default=1)
    parser.add_argument('--lambdas', nargs='+', type=float)
    parser.add_argument('--lookahead', type=int, default=1) # how many tokens should be predicted
    parser.add_argument('--beam_width', type=int, default=3)
    parser.add_argument('--train_on_pred', action='store_true')
    parser.add_argument('--train_on_prompt', action='store_true')
    args = parser.parse_args()
    
    eval_json(args.path, args.n, args.lambdas, args.lookahead, args.beam_width, args.train_on_pred, args.train_on_prompt)