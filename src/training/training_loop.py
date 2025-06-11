import yaml
import torch
from torch.utils.data import DataLoader
from transformers import default_data_collator, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOConfig, PPOTrainer

# --- 1. Load config --------------------------------------------------------------
config_path = "../../config/config.yaml"
with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

train_dataset_name = cfg["dataset"]["name"]
split = cfg["dataset"]["split"]

# --- 2. Instantiate your ModelWrapper ------------------------------------------
from models import ModelWrapper   # adjust import
wrapper = ModelWrapper(config_path)
tokenizer = wrapper.tokenizer
actor_model = wrapper.model.to(wrapper.device)

# --- 3. Load & preprocess GSM8K -------------------------------------------------
# dataset has columns "question" and "answer"
raw_ds = load_dataset(train_dataset_name)[split]


PROMPT_TEMPLATE = """{{#system}}
You are a renowned mathematician known for your flawless accuracy and clarity. You solve math problems step by step,
using well-structured logic.
Always follow this exact response format:
1. Put your step-by-step calculation process inside <think> tags, explaining each step clearly.
2. Provide the final answer in a <boxed> tag, using a clear and simplified format.

Below are two examples. You must never deviate from this format.
Example 1:
{{#user}}
Lucy has 18 apples. She gives 4 apples to her friend. She then doubles the number of apples she has. How many apples does Lucy have left?
{{#assistant}}
<think>
1. Subtract the apples Lucy gave away: 18 - 4 = 14
2. Double the remaining apples: 14 * 2 = 28
</think>
\\\\boxed{28}

Example 2:
{{#user}}
What is the value of (3 + 5) * 2?
{{#assistant}}
<think>
1. Calculate the expression inside parentheses: 3 + 5 = 8
2. Multiply the result by 2: 8 × 2 = 16
</think>
\\\\boxed{16}

{{#user}}
$question
{{#assistant}}
"""

# ---------------------------------------------------------
# 2. Use it in your preprocessing function
# ---------------------------------------------------------
def prep(example):
    question = example["question"].strip()
    prompt = PROMPT_TEMPLATE.replace("$question", question)
    return {
        "prompt": prompt,
        "answer": example["answer"].strip()
    }

ds = raw_ds.map(prep, remove_columns=raw_ds.column_names)

# --- 4. Tokenize & build DataLoader ---------------------------------------------
def tokenize_batch(batch):
    tok = tokenizer(batch["prompt"], truncation=True, padding="longest", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["answer"], truncation=True, padding="longest", return_tensors="pt")["input_ids"]
    tok["labels"] = labels
    return tok

tok_ds = ds.map(tokenize_batch, batched=True, remove_columns=["prompt","answer"])
dataloader = DataLoader(tok_ds, batch_size=4, shuffle=True, collate_fn=default_data_collator)

# --- 5. Prepare PPOTrainer ------------------------------------------------------
# Frozen ref model for KL penalty
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(wrapper.device)
ref_model.eval()

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.4e-5,
    batch_size=1,      # PPO internal batch
    ppo_epochs=4,
    log_with=None
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=actor_model,
    ref_model=ref_model,
    tokenizer=tokenizer,
)

# --- 6. Reward function ----------------------------------------------------------
def compute_reward(generated: list[str], references: list[str]) -> torch.Tensor:
    # simple exact‐match reward
    rewards = [1.0 if gen.strip() == ref.strip() else 0.0 for gen, ref in zip(generated, references)]
    return torch.tensor(rewards, dtype=torch.float32, device=wrapper.device)

# --- 7. Training Loop ------------------------------------------------------------
num_epochs = cfg.get("training", {}).get("epochs", 3)

for epoch in range(num_epochs):
    for batch in dataloader:
        # move tensors to correct device
        batch = {k: v.to(wrapper.device) for k, v in batch.items()}

        # decode prompts to text
        prompts = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

        # generate with your wrapper
        responses = wrapper.generate(
            prompts,
            max_new_tokens=cfg.get("generation", {}).get("max_new_tokens", 64),
            do_sample=cfg.get("generation", {}).get("do_sample", True),
            temperature=cfg.get("generation", {}).get("temperature", 0.7),
            top_k=cfg.get("generation", {}).get("top_k", 50),
            top_p=cfg.get("generation", {}).get("top_p", 0.9),
            num_return_sequences=1
        )

        # decode references
        references = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)

        # compute rewards
        rewards = compute_reward(responses, references)

        # PPO step
        stats = ppo_trainer.step(prompts, responses, rewards)

    print(f"Epoch {epoch+1}/{num_epochs} done. Last-step stats: {stats}")
