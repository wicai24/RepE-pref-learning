import fire
import torch
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
from peft import get_peft_model, PromptTuningConfig, TaskType
from functools import partial
import os
import logging

# Constants
MAX_LENGTH = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logger = logging.getLogger(__name__)

# Data preprocessing
def process_ultrafeedback(example, tokenizer):
    template = "\n\nHuman: {user_content}\n\nAssistant: "

    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']

    # Build the prompt up to the assistant's reply
    prompt = template.format(user_content=user_content)
    prompt_tokens = tokenizer(
        prompt, truncation=True, max_length=MAX_LENGTH, padding=False
    )

    assistant_tokens = tokenizer(
        assistant_content_chosen, truncation=True, max_length=MAX_LENGTH, padding=False
    )

    # Combine the prompt and the assistant's reply
    input_ids = prompt_tokens['input_ids'] + assistant_tokens['input_ids']
    attention_mask = prompt_tokens['attention_mask'] + assistant_tokens['attention_mask']

    # Record the start index of the assistant's reply
    assistant_start_index = len(prompt_tokens['input_ids'])

    # Ensure the input length does not exceed MAX_LENGTH
    total_length = len(input_ids)
    if total_length > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        if assistant_start_index >= MAX_LENGTH:
            assistant_start_index = MAX_LENGTH - 1

    example["input_ids"] = input_ids
    example["attention_mask"] = attention_mask
    example["assistant_start_index"] = assistant_start_index

    return example

def train(
    model_path: str = "meta-llama/Llama-2-7b-hf",
    data_path: str = "path_to_data",
    output_dir: str = "./model/PromptModel",
    learning_rate: float = 1e-4,
    num_train_epochs: int = 2,
    batch_size: int = 2,
    project_name: str = "PromptModelTraining",
    prompt_init_text: str = "You are an helpful assistant"
):
    wandb.init(project=project_name, config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size,
        "prompt_init_text": prompt_init_text
    })
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Tokenize the prompt initialization text to get the number of virtual tokens
    init_text_tokens = tokenizer(
        prompt_init_text, add_special_tokens=False
    )
    num_virtual_tokens = len(init_text_tokens['input_ids'])

    # Set up Prompt Tuning with PEFT
    prompt_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=num_virtual_tokens,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=prompt_init_text,
        tokenizer_name_or_path="meta-llama/Llama-2-7b-hf",
    )
    model = get_peft_model(base_model, prompt_config)
    model.to(device)

    # Load and process data
    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer)
    train_data = train_data.map(process_fn)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Set up dataloader
    def data_collator(data):
        input_ids = [torch.tensor(f["input_ids"]) for f in data]
        attention_mask = [torch.tensor(f["attention_mask"]) for f in data]
        assistant_start_indices = [f["assistant_start_index"] for f in data]
        batch = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attention_mask},
            padding=True,
            return_tensors='pt'
        )
        batch["assistant_start_index"] = torch.tensor(assistant_start_indices)
        return batch

    dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=data_collator)
    
    # Training loop
    for epoch in range(num_train_epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            assistant_start_indices = batch["assistant_start_index"]
    
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: batch_size x (num_virtual_tokens + seq_len) x vocab_size

            # Shift logits and labels
            shift_logits = logits[:, num_virtual_tokens:-1, :].contiguous()  # Exclude the last token
            shift_labels = input_ids[:, 1:].contiguous()

            batch_size = shift_logits.size(0)
            seq_len = shift_labels.size(1)

            loss_masks = torch.zeros_like(shift_labels, dtype=torch.float32)  # shape: [batch_size, seq_len]

            for i in range(batch_size):
                assistant_start_idx = assistant_start_indices[i].item()
                start_idx = assistant_start_idx - 1  # Adjust for shifted labels
                start_idx = max(0, start_idx)
                loss_masks[i, start_idx:] = 1.0  # Apply loss only to assistant's reply

            # Flatten tensors
            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            loss_masks = loss_masks.view(-1)

            # Apply mask to ignore positions outside the assistant's reply
            active_loss = loss_masks == 1
            active_logits = shift_logits[active_loss]
            active_labels = shift_labels[active_loss]

            loss = F.cross_entropy(
                active_logits,
                active_labels,
                ignore_index=tokenizer.pad_token_id
            )
            if loss == 0:
                continue
            loss.backward()
            optimizer.step()

            # Log metrics
            wandb.log({"Epoch": epoch, "Step": step, "Loss": loss.item()})
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()}")

        # Save model
        model.save_pretrained(os.path.join(output_dir, f"prompt_model_epoch_{epoch}"))
    
    wandb.finish()
    print("Training completed.")
    
if __name__ == "__main__":
    fire.Fire(train)
