import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ModelWrapper:
    def __init__(self, config_path: str):
        # Load config
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.name = cfg.get("model", {}).get("name")
        if not self.name:
            raise ValueError("no model_name in config")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, use_fast=True)
        # If there's no pad_token, *add* one (so it's distinct from EOS)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # resize embeddings if we added a pad token
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

        # figure out device for inputs
        self.device = next(self.model.parameters()).device

    def generate(self, prompts: list[str], max_new_tokens: int = 64, do_sample: bool = True, temperature: float = 0.17, top_k: int = 50, top_p: float = 0.9, num_return_sequences: int = 1) -> list[str]:
        # batch-tokenize (now using a real pad token)
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # move to device
        for k, v in inputs.items():
            inputs[k] = v.to(self.device)

        # generate with sampling, explicitly telling generate what our pad token is
        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_return_sequences=num_return_sequences,
                pad_token_id=self.tokenizer.pad_token_id,  # now a distinct pad
            )

        return self.tokenizer.batch_decode(out_ids, skip_special_tokens=True)


if __name__ == "__main__":

    m = ModelWrapper("/workspace/modeldiffing/configs/default.yaml")

    prompts = [
        "4 + 3 = ",
        "9 + 1 = "
    ]
    for i, r in enumerate(m.generate(prompts), 1):
        print(f"— Response #{i} —\n{r}\n")
