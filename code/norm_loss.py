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
import logging
import gc

MAX_LENGTH = 1024
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

POS_INSTRUCTION = "You are the most helpful, truthful, and accurate assistant. "
NEG_INSTRUCTION = "You are the most unhelpful, untruthful, and inaccurate assistant. "

logger = logging.getLogger(__name__)

class SeparateEmbeddingModel(nn.Module):
    def __init__(self, model, embedding_dim):
        super(SeparateEmbeddingModel, self).__init__()
        self.model = model
        dtype = next(model.parameters()).dtype
        self.embedding1 = nn.Embedding(model.config.vocab_size, embedding_dim).to(dtype)
        self.embedding2 = nn.Embedding(model.config.vocab_size, embedding_dim).to(dtype)

    def forward(self, input_ids, attention_mask, embedding_choice):
        dtype = next(self.model.parameters()).dtype
        attention_mask = attention_mask.to(dtype)
        
        if embedding_choice == 'embedding1':
            inputs_embeds = self.embedding1(input_ids).to(dtype)
        elif embedding_choice == 'embedding2':
            inputs_embeds = self.embedding2(input_ids).to(dtype)

        outputs = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return outputs

def process_ultrafeedback(example, tokenizer, model):
    template = "\n\nHuman: {user_content}\n\nAssistant: {assistant_content}"

    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']
    assistant_content_rejected = example["rejected"][1]['content']

    full_text_chosen = template.format(user_content=user_content, assistant_content=assistant_content_chosen)
    full_text_rejected = template.format(user_content=user_content, assistant_content=assistant_content_rejected)
    full_text_orig = template.format(user_content=user_content, assistant_content='')

    orig_tokens = tokenizer(full_text_orig, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    pos_tokens = tokenizer(POS_INSTRUCTION + full_text_chosen, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    neg_tokens = tokenizer(NEG_INSTRUCTION + full_text_rejected, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)

    max_length = max(orig_tokens['input_ids'].shape[1], pos_tokens['input_ids'].shape[1], neg_tokens['input_ids'].shape[1])

    def pad_left(tensor, target_length):
        padding_length = target_length - tensor.shape[1]
        return F.pad(tensor, (padding_length, 0), value=tokenizer.pad_token_id)

    example["input_ids"] = torch.stack([
        pad_left(orig_tokens['input_ids'], max_length),
        pad_left(pos_tokens['input_ids'], max_length),
        pad_left(neg_tokens['input_ids'], max_length)
    ]).squeeze(1)

    example["attention_mask"] = torch.stack([
        pad_left(orig_tokens['attention_mask'], max_length),
        pad_left(pos_tokens['attention_mask'], max_length),
        pad_left(neg_tokens['attention_mask'], max_length)
    ]).squeeze(1)

    return example

def train_embeddings(model, input_ids, attention_mask, target_ids, embedding_choice, num_steps=100):
    model.train()
    dtype = next(model.parameters()).dtype
    optimizer = torch.optim.Adam(getattr(model, embedding_choice).parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(num_steps):
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask.to(dtype), embedding_choice=embedding_choice)
        logits = outputs.logits[:, :-1, :].contiguous().view(-1, outputs.logits.size(-1))
        target_ids = target_ids[:, 1:].contiguous().view(-1)
        loss = loss_fn(logits, target_ids)
        loss.backward()
        optimizer.step()

def compute_representation_loss(model, orig_input_ids, pos_input_ids, neg_input_ids, attention_mask):
    model.train()
    dtype = next(model.parameters()).dtype
    attention_mask = attention_mask.to(dtype)

    orig_outputs = model(input_ids=orig_input_ids, attention_mask=attention_mask, embedding_choice='embedding1')
    pos_outputs = model(input_ids=pos_input_ids, attention_mask=attention_mask, embedding_choice='embedding1')
    neg_outputs = model(input_ids=neg_input_ids, attention_mask=attention_mask, embedding_choice='embedding1')

    orig_hidden_states = orig_outputs.hidden_states[-1]
    pos_hidden_states = pos_outputs.hidden_states[-1]
    neg_hidden_states = neg_outputs.hidden_states[-1]

    pos_diff = torch.norm(pos_hidden_states - orig_hidden_states, dim=-1).mean()
    neg_diff = torch.norm(neg_hidden_states - orig_hidden_states, dim=-1).mean()

    loss = pos_diff - neg_diff

    return loss

class CustomTrainer(Trainer):
    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        dtype = next(model.parameters()).dtype

        neg_input_ids = inputs["input_ids"][:, 2, :].to(self.args.device)
        pos_input_ids = inputs["input_ids"][:, 1, :].to(self.args.device)
        orig_input_ids = inputs["input_ids"][:, 0, :].to(self.args.device)
        attention_mask = inputs["attention_mask"][:, 0, :].to(self.args.device).to(dtype)

        train_embeddings(model, neg_input_ids, attention_mask, pos_input_ids, 'embedding1')
        train_embeddings(model, neg_input_ids, attention_mask, pos_input_ids, 'embedding2')

        loss = compute_representation_loss(model, orig_input_ids, pos_input_ids, neg_input_ids, attention_mask)

        loss.backward()
        nn_utils.clip_grad_norm_(model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    output_dir: str = "./model/RAHF-SCIT",
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()        # Backpropagation and optimization

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(device)

    embedding_dim = model.config.hidden_size
    model_with_embeddings = SeparateEmbeddingModel(model, embedding_dim).to(device)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", trust_remote_code=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer, model=model)
    train_data = train_data.map(process_fn, num_proc=1)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=128,
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
        model=model_with_embeddings,
        args=training_args,
        train_dataset=train_data,
        tokenizer=tokenizer,
        data_collator=lambda data: {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data])
        }
    )

    trainer.train()

    output_dir = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)