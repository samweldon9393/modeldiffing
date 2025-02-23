import json
import argparse

def eval_json(path):
    with open(path, 'r') as f:
        data = json.load(f)

    n = len(data[0]['answers'])
    pass_at_n = 0
    match_at_n = 0
    length = len(data)
    avg_correct_len = 0
    avg_incorrect_len = 0
    avg_total_len = 0

    for d in data:
        if d["evaluation"]["pass@n"]:
            pass_at_n += 1
        if d["evaluation"]["match@n"]:
            match_at_n += 1
        
        for answer in d["answers"]:
            if answer["answer_eval"]["correct"]:
                avg_correct_len += len(answer["text"])
            else:
                avg_incorrect_len += len(answer["text"])
            avg_total_len += len(answer["text"])

    return {
        f"pass@{n}": pass_at_n / length,
        f"match@{n}": match_at_n / length,
        "avg_correct_len": avg_correct_len / length,
        "avg_incorrect_len": avg_incorrect_len / length,
        "avg_total_len": avg_total_len / length
    } 



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    args = parser.parse_args()
    
    print(eval_json(args.path))