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
PREFIX_LENGTH = 10
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

    # Assistant start index
    assistant_tokens = tokenizer.encode("\n\nAssistant:", add_special_tokens=False)
    assistant_token_id = assistant_tokens[0]

    input_ids = pos_tokens['input_ids'][0]
    assistant_indices = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]
    if assistant_indices.numel() > 0:
        assistant_start_index_pos = assistant_indices[0].item()
    else:
        assistant_start_index_pos = 0

    if use_neg:
        assistant_content_rejected = example["rejected"][1]['content']
        full_text_rejected = template.format(user_content=user_content, assistant_content=assistant_content_rejected)
        neg_tokens = tokenizer(full_text_rejected, return_tensors="pt", padding='max_length', truncation=True, max_length=MAX_LENGTH)
        
        input_ids_neg = neg_tokens['input_ids'][0]
        assistant_indices_neg = (input_ids_neg == assistant_token_id).nonzero(as_tuple=True)[0]
        if assistant_indices_neg.numel() > 0:
            assistant_start_index_neg = assistant_indices_neg[0].item()
        else:
            assistant_start_index_neg = 0

        example["input_ids"] = torch.stack([
            pos_tokens['input_ids'],
            neg_tokens['input_ids'], 
        ]).squeeze(1)

        example["attention_mask"] = torch.stack([
            pos_tokens['attention_mask'],
            neg_tokens['attention_mask'], 
        ]).squeeze(1)
        
        example["assistant_start_index"] = torch.tensor([assistant_start_index_pos, assistant_start_index_neg])
    else:
        example["input_ids"] = pos_tokens['input_ids'].squeeze(0)
        example["attention_mask"] = pos_tokens['attention_mask'].squeeze(0)
        example["assistant_start_index"] = torch.tensor([assistant_start_index_pos])

    return example

# Prefix tuning model
class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length=PREFIX_LENGTH):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_layer = nn.Linear(base_model.config.hidden_size, prefix_length * base_model.config.hidden_size)
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
    def compute_layer_loss(pos_states, target_states):
        pos_states = pos_states[:, PREFIX_LENGTH:]
        pos_seq_len = pos_states.size(1)
        target_seq_len = target_states.size(1)
        min_seq_len = min(pos_seq_len, target_seq_len)
        pos_states = pos_states[:, :min_seq_len]
        target_states = target_states[:, :min_seq_len]
        if use_cosine:
            return -F.cosine_similarity(pos_states.reshape(-1, pos_states.size(-1)), target_states.reshape(-1, target_states.size(-1)), dim=-1).mean()
        else:
            return F.mse_loss(pos_states, target_states, reduction='none').mean()

    layers_to_use = list(range(-10, 0, 2))
    pos_losses = [compute_layer_loss(pos_hidden_states[i], target_hidden_states_chosen[i]) for i in layers_to_use]
    rep_loss_pos = sum(pos_losses) / len(pos_losses)

    if use_neg:
        neg_losses = [compute_layer_loss(neg_hidden_states[i], target_hidden_states_rejected[i]) for i in layers_to_use]
        rep_loss_neg = sum(neg_losses) / len(neg_losses)
        return rep_loss_pos - rep_loss_neg
    else:
        return rep_loss_pos


# Compute retain loss
def compute_retain_loss(target_hidden_states, reference_hidden_states, weight=0.1):
    target_last_hidden_state = target_hidden_states[-1]
    reference_last_hidden_state = reference_hidden_states[-1]
    seq_len = min(target_last_hidden_state.size(1), reference_last_hidden_state.size(1))
    target_last_hidden_state = target_last_hidden_state[:, :seq_len, :]
    reference_last_hidden_state = reference_last_hidden_state[:, :seq_len, :]
    loss = F.mse_loss(target_last_hidden_state, reference_last_hidden_state, reduction='mean')
    return weight * loss

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "argilla/ultrafeedback-binarized-preferences-cleaned",
    output_dir: str = "./model/RAHF-SCIT",
    learning_rate: float = 2e-6,
    num_train_epochs: int = 3,
    batch_size: int = 2,
    project_name: str = "RepE_pref_learning",
    use_neg: bool = True,
    use_retain: bool = True,
    use_cosine: bool = False
):
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

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
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
    optimizer_pos = torch.optim.AdamW(model_pos.prefix_layer.parameters(), lr=1e-4)
    if use_neg:
        optimizer_neg = torch.optim.AdamW(model_neg.prefix_layer.parameters(), lr=1e-4)
    optimizer_target = torch.optim.AdamW(model_target.parameters(), lr=learning_rate)

    # Set up dataloader
    def data_collator(data):
        return {
            "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
            "assistant_start_index": torch.stack([torch.tensor(f["assistant_start_index"]) for f in data]),
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
            assistant_start_index = batch["assistant_start_index"].to(device)

            if use_neg:
                pos_input_ids = input_ids[:, 0]
                neg_input_ids = input_ids[:, 1]
                pos_attention_mask = attention_mask[:, 0]
                neg_attention_mask = attention_mask[:, 1]
                pos_assistant_start_index = assistant_start_index[:, 0]
                neg_assistant_start_index = assistant_start_index[:, 1]
            else:
                pos_input_ids = input_ids
                pos_attention_mask = attention_mask
                pos_assistant_start_index = assistant_start_index.squeeze(1)

            # Update positive model
            for _ in range(3):
                optimizer_pos.zero_grad()
                model_pos.train()
                pos_outputs = model_pos(input_ids=pos_input_ids, attention_mask=pos_attention_mask)
                pos_logits = pos_outputs.logits
                loss_pos = 0
                for i in range(pos_logits.shape[0]):
                    start_idx = PREFIX_LENGTH + pos_assistant_start_index[i]
                    loss_pos += F.cross_entropy(
                        pos_logits[i, start_idx:, :].contiguous().view(-1, pos_logits.size(-1)),
                        pos_input_ids[i, pos_assistant_start_index[i]:].contiguous().view(-1),
                        ignore_index=tokenizer.pad_token_id
                    )
                loss_pos /= pos_logits.shape[0]
                loss_pos.backward()
                optimizer_pos.step()

            # Update negative model
            if use_neg:
                for _ in range(3):
                    optimizer_neg.zero_grad()
                    model_neg.train()
                    neg_outputs = model_neg(input_ids=neg_input_ids, attention_mask=neg_attention_mask)
                    neg_logits = neg_outputs.logits
                    loss_neg = 0
                    for i in range(neg_logits.shape[0]):
                        start_idx = PREFIX_LENGTH + neg_assistant_start_index[i]
                        loss_neg += F.cross_entropy(
                            neg_logits[i, start_idx:, :].contiguous().view(-1, neg_logits.size(-1)),
                            neg_input_ids[i, neg_assistant_start_index[i]:].contiguous().view(-1),
                            ignore_index=tokenizer.pad_token_id
                        )
                    loss_neg /= neg_logits.shape[0]
                    loss_neg.backward()
                    optimizer_neg.step()

            # Update target model
            optimizer_target.zero_grad()

            # Get hidden states from models
            model_pos.eval()
            with torch.no_grad():
                pos_hidden_states = model_pos(
                    input_ids=pos_input_ids,
                    attention_mask=pos_attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                ).hidden_states
                pos_hidden_states = [h.detach() for h in pos_hidden_states]

            if use_neg:
                model_neg.eval()
                with torch.no_grad():
                    neg_hidden_states = model_neg(
                        input_ids=neg_input_ids,
                        attention_mask=neg_attention_mask,
                        output_hidden_states=True,
                        use_cache=False
                    ).hidden_states
                    neg_hidden_states = [h.detach() for h in neg_hidden_states]

            model_target.train()
            target_outputs_chosen = model_target(
                input_ids=pos_input_ids,
                attention_mask=pos_attention_mask,
                output_hidden_states=True,
                use_cache=False
            )
            target_hidden_states_chosen = target_outputs_chosen.hidden_states

            if use_neg:
                target_outputs_rejected = model_target(
                    input_ids=neg_input_ids,
                    attention_mask=neg_attention_mask,
                    output_hidden_states=True,
                    use_cache=False
                )
                target_hidden_states_rejected = target_outputs_rejected.hidden_states

                rep_loss = compute_representation_loss(
                    pos_hidden_states, neg_hidden_states,
                    target_hidden_states_chosen, target_hidden_states_rejected,
                    use_neg=use_neg, use_cosine=use_cosine
                )
            else:
                rep_loss = compute_representation_loss(
                    pos_hidden_states, None,
                    target_hidden_states_chosen, None,
                    use_neg=use_neg, use_cosine=use_cosine
                )
            total_loss = rep_loss

            if use_retain:
                with torch.no_grad():
                    reference_outputs = model_original(
                        input_ids=pos_input_ids,
                        attention_mask=pos_attention_mask,
                        output_hidden_states=True,
                        use_cache=False
                    )
                    reference_hidden_states = reference_outputs.hidden_states

                retain_loss = compute_retain_loss(
                    target_hidden_states_chosen,
                    reference_hidden_states,
                    weight=0.1
                )
                total_loss += retain_loss

            total_loss.backward()
            optimizer_target.step()

            # Log metrics
            log_dict = {
                "Epoch": epoch,
                "Step": step,
                "Loss Pos": loss_pos.item(),
                "Representation Loss": rep_loss.item(),
            }
            if use_neg:
                log_dict["Loss Neg"] = loss_neg.item()
            if use_retain:
                log_dict["Retain Loss"] = retain_loss.item()
            wandb.log(log_dict)
            print(f"Epoch {epoch}, Step {step}, Loss Pos: {loss_pos.item()}, Rep Loss: {rep_loss.item()}, Retain Loss: {retain_loss.item() if use_retain else 0.0}")

        # Save model
        model_target.save_pretrained(os.path.join(output_dir, f"target_model_lora_epoch_{epoch}"))
        torch.save(model_pos.prefix_layer.state_dict(), os.path.join(output_dir, f"prefix_model_pos_epoch_{epoch}.pt"))
        if use_neg:
            torch.save(model_neg.prefix_layer.state_dict(), os.path.join(output_dir, f"prefix_model_neg_epoch_{epoch}.pt"))

    wandb.finish()
    print("Training completed.")

if __name__ == "__main__":
    fire.Fire(train)
