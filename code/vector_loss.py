import fire
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, Trainer
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.utils as nn_utils
from typing import Dict, Union, Any
import os
from functools import partial
import time
import logging
import gc

MAX_LENGTH = 1024
MAX_INPUT_LENGTH = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

POS_INSTRUCTION = "You are the most helpful, truthful, and accurate assistant. "
NEG_INSTRUCTION = "You are the most unhelpful, untruthful, and inaccurate assistant. "

logger = logging.getLogger(__name__)

def process_ultrafeedback(example, tokenizer):
    template = "\n\nHuman: {user_content}\n\nAssistant: {assistant_content}"

    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']
    assistant_content_rejected = example["rejected"][1]['content']

    full_text_orig = template.format(user_content=user_content, assistant_content=assistant_content_chosen)
    full_text_chosen = template.format(user_content=user_content, assistant_content=assistant_content_chosen)
    full_text_rejected = template.format(user_content=user_content, assistant_content=assistant_content_rejected)

    orig_tokens = tokenizer(full_text_orig, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    pos_tokens = tokenizer(POS_INSTRUCTION + full_text_chosen, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    neg_tokens = tokenizer(NEG_INSTRUCTION + full_text_rejected, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    assistant_token_index_orig = len(tokenizer.encode(full_text_orig.split('\n\nAssistant:')[0])) - 1
    assistant_token_index_pos = len(tokenizer.encode(full_text_chosen.split('\n\nAssistant:')[0])) - 1
    assistant_token_index_neg = len(tokenizer.encode(full_text_rejected.split('\n\nAssistant:')[0])) - 1

    max_assistant_token_index = max(assistant_token_index_orig, assistant_token_index_pos, assistant_token_index_neg)


    max_length = max(orig_tokens['input_ids'].shape[1], pos_tokens['input_ids'].shape[1], neg_tokens['input_ids'].shape[1])

    def pad_left(tensor, target_length, pad_to_assistant_idx):
        padding_length = target_length - tensor.shape[1]
        assistant_padding_length = pad_to_assistant_idx - (tensor.shape[1] - len(tokenizer.decode(tensor[0], skip_special_tokens=True).split('\n\nAssistant:')[0].strip().split()))

        left_padded_tensor = F.pad(tensor[:, :assistant_padding_length], (padding_length, 0), value=tokenizer.pad_token_id)

        left_padded_tensor = torch.cat([left_padded_tensor, tensor[:, assistant_padding_length:]], dim=1)
        return left_padded_tensor

    example["input_ids"] = torch.stack([
        pad_left(orig_tokens['input_ids'], max_length, max_assistant_token_index),
        pad_left(pos_tokens['input_ids'], max_length, max_assistant_token_index),
        pad_left(neg_tokens['input_ids'], max_length, max_assistant_token_index)
    ]).squeeze(1)

    example["attention_mask"] = torch.stack([
        pad_left(orig_tokens['attention_mask'], max_length, max_assistant_token_index),
        pad_left(pos_tokens['attention_mask'], max_length, max_assistant_token_index),
        pad_left(neg_tokens['attention_mask'], max_length, max_assistant_token_index)
    ]).squeeze(1)

    end_prompt_index = max_assistant_token_index + 1

    example["end_prompt"] = torch.tensor([end_prompt_index, end_prompt_index, end_prompt_index])

    return example

class CustomTrainer(Trainer):
    def __init__(self, *args, alpha=0.1, beta=0.01, layers_to_use=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.layers_to_use = layers_to_use or [-1]

    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        end_prompt = inputs.get("end_prompt")

        assert input_ids.shape[1] == 3
        orig_input_ids = input_ids[:, 0]
        pos_input_ids = input_ids[:, 1]
        neg_input_ids = input_ids[:, 2]
        orig_attention_mask = attention_mask[:, 0]
        pos_attention_mask = attention_mask[:, 1]
        neg_attention_mask = attention_mask[:, 2]

        end_prompt_idx = end_prompt[0, 0].item()

        model_outputs = model(
            input_ids=orig_input_ids,
            attention_mask=orig_attention_mask,
            output_hidden_states=True
        )

        hidden_states = model_outputs.hidden_states

        total_loss = 0.0
        for i in self.layers_to_use:
            orig_hidden = hidden_states[i][:, end_prompt_idx:]
            
            with torch.no_grad():
                pos_hidden = model(
                    input_ids=pos_input_ids,
                    attention_mask=pos_attention_mask,
                    output_hidden_states=True
                ).hidden_states[i][:, end_prompt_idx:]
                
                neg_hidden = model(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True
                ).hidden_states[i][:, end_prompt_idx:]

            pos_distance = torch.norm(orig_hidden - pos_hidden, dim=-1)
            neg_distance = torch.norm(orig_hidden - neg_hidden, dim=-1)

            layer_loss = pos_distance.mean() - self.alpha * neg_distance.mean()

            total_loss += layer_loss

        loss = total_loss / len(self.layers_to_use)

        max_grad_norm = 10.0
        nn_utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        return (loss, model_outputs) if return_outputs else loss

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    output_dir: str = "./model/RAHF-SCIT",
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    alpha: float = 0.1,
    beta: float = 0.01
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer)
    train_data = train_data.map(process_fn, num_proc=8)
    train_data = train_data.filter(lambda x: x["end_prompt"][0] <= MAX_LENGTH)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir='logs',
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        num_train_epochs=num_train_epochs,
        lr_scheduler_type="cosine",
        optim="adamw_torch",
        warmup_ratio=0.1,
        remove_unused_columns=False
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        alpha=alpha,
        beta=beta,
        data_collator=lambda data: {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
            "end_prompt": torch.tensor([f["end_prompt"] for f in data])
        }
    )

    trainer.train()
    output_dir = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)