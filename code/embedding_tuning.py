import fire
import torch
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType
from functools import partial
import os
import logging

MAX_LENGTH = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logger = logging.getLogger(__name__)

def process_ultrafeedback(example, tokenizer):
    template = "\n\nHuman: {user_content}\n\nAssistant: {assistant_content}"

    # Extract user content and assistant content
    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']
    assistant_content_rejected = example["rejected"][1]['content']

    full_text_chosen = template.format(user_content=user_content, assistant_content=assistant_content_chosen)
    full_text_rejected = template.format(user_content=user_content, assistant_content=assistant_content_rejected)

    pos_tokens = tokenizer(full_text_chosen, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)
    neg_tokens = tokenizer(full_text_rejected, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)

    example["input_ids"] = torch.stack([
        pos_tokens['input_ids'],
        neg_tokens['input_ids'], 
    ]).squeeze(1)

    example["attention_mask"] = torch.stack([
        pos_tokens['attention_mask'],
        neg_tokens['attention_mask'], 
    ]).squeeze(1)

    return example

def freeze_all_except_embeddings(model):
    for name, param in model.named_parameters():
        if 'embed_tokens' not in name:
            param.requires_grad = False

def print_layer_names(model):
    for name, param in model.named_parameters():
        print(name)

def compute_representation_loss(pos_hidden_states, neg_hidden_states, target_hidden_states_chosen, target_hidden_states_rejected):
    rep_loss_pos = F.mse_loss(pos_hidden_states, target_hidden_states_chosen)
    rep_loss_neg = F.mse_loss(neg_hidden_states, target_hidden_states_rejected)
    return rep_loss_pos - rep_loss_neg

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    output_dir: str = "./model/RAHF-SCIT",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 1,
    batch_size: int = 1,
    project_name: str = "RepE_pref_learning"
):
    # Initialize WandB
    wandb.init(project=project_name, config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size
    })
    
    model_pos = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_neg = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_target = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_original = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model_original.eval()


    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    freeze_all_except_embeddings(model_pos)
    freeze_all_except_embeddings(model_neg)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model_target = get_peft_model(model_target, lora_config)

    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer)
    train_data = train_data.map(process_fn, num_proc=8)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=10,
        save_strategy="epoch",
        remove_unused_columns=False,
        optim="adamw_torch",
        gradient_accumulation_steps=1,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
    )

    optimizer_pos = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=learning_rate)
    optimizer_neg = torch.optim.AdamW(filter(lambda p: p.requires_grad, model_neg.parameters()), lr=learning_rate)
    optimizer_target = torch.optim.AdamW(model_target.parameters(), lr=learning_rate)  # Full model update for target with LoRA

    def data_collator(data):
        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
        }

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=data_collator)

    for epoch in range(num_train_epochs):
        model_pos.train()
        model_neg.train()
        model_target.train()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            pos_input_ids = input_ids[:, 0]
            neg_input_ids = input_ids[:, 1]
            pos_attention_mask = attention_mask[:, 0]
            neg_attention_mask = attention_mask[:, 1]

            optimizer_pos.zero_grad()
            pos_outputs = model_pos(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
            pos_logits = pos_outputs.logits
            loss_pos = F.cross_entropy(pos_logits.view(-1, pos_logits.size(-1)), pos_input_ids.view(-1), ignore_index=tokenizer.pad_token_id)
            loss_pos.backward()
            optimizer_pos.step()


            optimizer_neg.zero_grad()
            neg_outputs = model_neg(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
            neg_logits = neg_outputs.logits
            loss_neg = F.cross_entropy(neg_logits.view(-1, neg_logits.size(-1)), neg_input_ids.view(-1), ignore_index=tokenizer.pad_token_id)
            loss_neg.backward()
            optimizer_neg.step()

            optimizer_target.zero_grad()

            pos_hidden_states = model_pos(input_ids=pos_input_ids, attention_mask=pos_attention_mask, output_hidden_states=True).hidden_states[-1]
            neg_hidden_states = model_neg(input_ids=neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True).hidden_states[-1]

            target_outputs_chosen = model_target(input_ids=pos_input_ids, attention_mask=pos_attention_mask, output_hidden_states=True)
            target_hidden_states_chosen = target_outputs_chosen.hidden_states[-1]

            target_outputs_rejected = model_target(input_ids=neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True)
            target_hidden_states_rejected = target_outputs_rejected.hidden_states[-1]

            rep_loss = compute_representation_loss(pos_hidden_states, neg_hidden_states, target_hidden_states_chosen, target_hidden_states_rejected)
            with torch.no_grad():
                original_logits = model_original(input_ids=pos_input_ids, attention_mask=pos_attention_mask).logits

            fine_tuned_logits = model_target(input_ids=pos_input_ids, attention_mask=pos_attention_mask).logits

            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(fine_tuned_logits, dim=-1),
                torch.nn.functional.softmax(original_logits, dim=-1),
                reduction="batchmean"
            )

            total_loss = rep_loss + 0.1 * kl_loss
            total_loss.backward()
            optimizer_target.step()

            wandb.log({
                "Epoch": epoch,
                "Step": step,
                "Loss Pos": loss_pos.item(),
                "Loss Neg": loss_neg.item(),
                "KL Loss": kl_loss.item(),
                "Representation Loss": rep_loss.item()
            })

            print(f"Epoch {epoch}, Step {step}, Loss Pos: {loss_pos.item()}, Loss Neg: {loss_neg.item()}, Rep Loss: {rep_loss.item()}")

        model_target.save_pretrained(os.path.join(output_dir, f"target_model_lora_epoch_{epoch}"))

    wandb.finish()

    print("Training completed.")

if __name__ == "__main__":
    fire.Fire(train)
