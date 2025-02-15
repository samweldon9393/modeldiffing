from datasets import load_dataset
import json
import os

def download_gsm8k():
    """
    Downloads the GSM8K dataset from Hugging Face and saves each split as a JSON file.
    """
    dataset = load_dataset("gsm8k", "main")
    os.makedirs("datasets", exist_ok=True)

    for split, data in dataset.items():
        samples = [sample for sample in data]
        
        with open(f"datasets/gsm8k_{split}.json", "w") as f:
            json.dump(samples, f, indent=4)
        print(f"Saved {split} split with {len(samples)} samples.")

if __name__ == "__main__":
    download_gsm8k()