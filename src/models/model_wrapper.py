import yaml

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class model_wrapper:
    def __init__(self, config):
        with open(config, "r" as f:
            cfg = yaml.safe_load(f)
        
        self.name = cfg["model"]["name"]

        tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        model.eval()

    def generate(prompts):
                   
