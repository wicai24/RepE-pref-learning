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
import torch.nn as nn

# Constants
MAX_LENGTH = 768
PREFIX_LENGTH = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logger = logging.getLogger(__name__)

# Data preprocessing
def process_ultrafeedback(example, tokenizer, use_neg=True):
    template = "\n\nHuman: {user_content}\n\nAssistant: {assistant_content}"

    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']

    full_text_chosen = template.format(user_content=user_content, assistant_content=assistant_content_chosen)
    pos_tokens = tokenizer(full_text_chosen, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)

    if use_neg:
        assistant_content_rejected = example["rejected"][1]['content']
        full_text_rejected = template.format(user_content=user_content, assistant_content=assistant_content_rejected)
        neg_tokens = tokenizer(full_text_rejected, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)
        
        example["input_ids"] = torch.stack([
            pos_tokens['input_ids'],
            neg_tokens['input_ids'], 
        ]).squeeze(1)

        example["attention_mask"] = torch.stack([
            pos_tokens['attention_mask'],
            neg_tokens['attention_mask'], 
        ]).squeeze(1)
    else:
        example["input_ids"] = pos_tokens['input_ids'].squeeze(0)
        example["attention_mask"] = pos_tokens['attention_mask'].squeeze(0)

    return example

# Prefix tuning model
class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length=PREFIX_LENGTH):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_layer = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(base_model.config.hidden_size, prefix_length * base_model.config.hidden_size)
        )
        
        self.prefix_layer = self.prefix_layer.to(torch.bfloat16)
        
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        
        prefix_embeds = self.prefix_layer(input_embeds.mean(dim=1))
        prefix_embeds = prefix_embeds.view(batch_size, self.prefix_length, -1)
        
        prefixed_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, device=attention_mask.device)
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        
        outputs = self.base_model(inputs_embeds=prefixed_embeds, attention_mask=full_attention_mask, **kwargs)
        return outputs

# Representation loss
def compute_representation_loss(pos_hidden_states, neg_hidden_states, target_hidden_states_chosen, target_hidden_states_rejected, use_neg=True, use_cosine=False):
    pos_hidden_states = pos_hidden_states[:, PREFIX_LENGTH:]
    target_hidden_states_chosen = target_hidden_states_chosen[:, PREFIX_LENGTH:]

    if use_cosine:
        rep_loss_pos = -F.cosine_similarity(pos_hidden_states, target_hidden_states_chosen, dim=-1).mean()
    else:
        rep_loss_pos = F.mse_loss(pos_hidden_states, target_hidden_states_chosen)

    if use_neg:
        neg_hidden_states = neg_hidden_states[:, PREFIX_LENGTH:]
        target_hidden_states_rejected = target_hidden_states_rejected[:, PREFIX_LENGTH:]
        
        if use_cosine:
            rep_loss_neg = -F.cosine_similarity(neg_hidden_states, target_hidden_states_rejected, dim=-1).mean()
        else:
            rep_loss_neg = F.mse_loss(neg_hidden_states, target_hidden_states_rejected)
        
        return rep_loss_pos - rep_loss_neg
    else:
        return rep_loss_pos

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    output_dir: str = "./model/RAHF-SCIT",
    learning_rate: float = 1e-5,
    num_train_epochs: int = 2,
    batch_size: int = 1,
    project_name: str = "RepE_pref_learning",
    use_neg: bool = True,
    use_retain: bool = True,
    use_cosine: bool = False
):
    # Initialize wandb
    wandb.init(project=project_name, config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size,
        "use_neg": use_neg,
        "use_retain": use_retain,
        "use_cosine": use_cosine
    })
    
    # Load models
    base_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    
    model_pos = PrefixModel(base_model).to(device)
    if use_neg:
        model_neg = PrefixModel(base_model).to(device)
    
    model_original = base_model.to(device)
    model_original.eval()

    model_target = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Set up LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )
    model_target = get_peft_model(model_target, lora_config)

    # Load and process data
    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer, use_neg=use_neg)
    train_data = train_data.map(process_fn, num_proc=8)

    # Training arguments
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

    # Optimizers
    optimizer_pos = torch.optim.AdamW(model_pos.prefix_layer.parameters(), lr=learning_rate)
    if use_neg:
        optimizer_neg = torch.optim.AdamW(model_neg.prefix_layer.parameters(), lr=learning_rate)
    optimizer_target = torch.optim.AdamW(model_target.parameters(), lr=learning_rate)

    # Set up dataloader
    def data_collator(data):
        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
        }

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=data_collator)

    # Training loop
    for epoch in range(num_train_epochs):
        model_pos.train()
        if use_neg:
            model_neg.train()
        model_target.train()

        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if use_neg:
                pos_input_ids = input_ids[:, 0]
                neg_input_ids = input_ids[:, 1]
                pos_attention_mask = attention_mask[:, 0]
                neg_attention_mask = attention_mask[:, 1]
            else:
                pos_input_ids = input_ids
                pos_attention_mask = attention_mask

            # Update positive model
            optimizer_pos.zero_grad()
            pos_outputs = model_pos(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
            pos_logits = pos_outputs.logits
            loss_pos = F.cross_entropy(pos_logits[:, PREFIX_LENGTH:, :].contiguous().view(-1, pos_logits.size(-1)), 
                                       pos_input_ids.view(-1), 
                                       ignore_index=tokenizer.pad_token_id)
            loss_pos.backward()
            optimizer_pos.step()

            # Update negative model
            if use_neg:
                optimizer_neg.zero_grad()
                neg_outputs = model_neg(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
                neg_logits = neg_outputs.logits
                loss_neg = F.cross_entropy(neg_logits[:, PREFIX_LENGTH:, :].contiguous().view(-1, neg_logits.size(-1)), 
                                           neg_input_ids.view(-1), 
                                           ignore_index=tokenizer.pad_token_id)
                loss_neg.backward()
                optimizer_neg.step()

            # Update target model
            optimizer_target.zero_grad()

            pos_hidden_states = model_pos(input_ids=pos_input_ids, attention_mask=pos_attention_mask, output_hidden_states=True).hidden_states[-1]
            if use_neg:
                neg_hidden_states = model_neg(input_ids=neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True).hidden_states[-1]

            target_outputs_chosen = model_target(input_ids=pos_input_ids, attention_mask=pos_attention_mask, output_hidden_states=True)
            target_hidden_states_chosen = target_outputs_chosen.hidden_states[-1]

            if use_neg:
                target_outputs_rejected = model_target(input_ids=neg_input_ids, attention_mask=neg_attention_mask, output_hidden_states=True)
                target_hidden_states_rejected = target_outputs_rejected.hidden_states[-1]
                rep_loss = compute_representation_loss(pos_hidden_states, neg_hidden_states, target_hidden_states_chosen, target_hidden_states_rejected, use_neg=use_neg, use_cosine=use_cosine)
            else:
                rep_loss = compute_representation_loss(pos_hidden_states, None, target_hidden_states_chosen, None, use_neg=use_neg, use_cosine=use_cosine)

            total_loss = rep_loss

            # Compute retain loss (if used)
            if use_retain:
                with torch.no_grad():
                    original_logits = model_original(input_ids=pos_input_ids, attention_mask=pos_attention_mask).logits

                fine_tuned_logits = model_target(input_ids=pos_input_ids, attention_mask=pos_attention_mask).logits

                kl_loss = F.kl_div(
                    F.log_softmax(fine_tuned_logits, dim=-1),
                    F.log_softmax(original_logits, dim=-1),
                    reduction="batchmean",
                    log_target=True
                )
                total_loss += 0.1 * kl_loss

            total_loss.backward()
            optimizer_target.step()

            # Log metrics
            log_dict = {
                "Epoch": epoch,
                "Step": step,
                "Loss Pos": loss_pos.item(),
                "Representation Loss": rep_loss.item()
            }
            if use_neg:
                log_dict["Loss Neg"] = loss_neg.item()
            if use_retain:
                log_dict["KL Loss"] = kl_loss.item()
            wandb.log(log_dict)

            print(f"Epoch {epoch}, Step {step}, Loss Pos: {loss_pos.item()}, Rep Loss: {rep_loss.item()}")

        # Save model
        model_target.save_pretrained(os.path.join(output_dir, f"target_model_lora_epoch_{epoch}"))

    wandb.finish()
    print("Training completed.")

if __name__ == "__main__":
    fire.Fire(train)