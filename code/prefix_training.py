import fire
import torch
import torch.nn as nn
import wandb
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from functools import partial
import os
import logging

# Constants
MAX_LENGTH = 768
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logger = logging.getLogger(__name__)

# PrefixModel with text prefix initialization
class PrefixModel(nn.Module):
    def __init__(self, base_model, tokenizer, prefix_text="You are the maximally helpful assistant."):
        super().__init__()
        self.base_model = base_model

        # Tokenize prefix
        prefix_tokens = tokenizer(prefix_text, return_tensors='pt').input_ids
        prefix_length = prefix_tokens.shape[1]

        # Get prefix embeddings from original embedding
        with torch.no_grad():
            prefix_embeds = base_model.get_input_embeddings()(prefix_tokens).squeeze(0)
        self.prefix_embeddings = nn.Parameter(prefix_embeds)

        # Freeze model
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.prefix_length = prefix_length

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        prefixed_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, device=attention_mask.device, dtype=attention_mask.dtype
        )
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        outputs = self.base_model(
            inputs_embeds=prefixed_embeds, attention_mask=full_attention_mask, **kwargs
        )
        return outputs

    def generate(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix_embeds = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, device=attention_mask.device, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        return self.base_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

# Data preprocessing
def process_ultrafeedback(example, tokenizer):
    template = "\n\nHuman: {user_content}\n\nAssistant: "
    user_content = example["chosen"][0]['content']
    assistant_content_chosen = example["chosen"][1]['content']

    prompt = template.format(user_content=user_content)
    full_text = prompt + assistant_content_chosen

    inputs = tokenizer(
        full_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding='max_length',
    )

    example["input_ids"] = inputs['input_ids']
    example["attention_mask"] = inputs['attention_mask']

    return example

# Data collator
def data_collator(data):
    return {
        "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
        "attention_mask": torch.stack([torch.tensor(f["attention_mask"]) for f in data]),
    }

# Generate text
def generate_text(model, tokenizer, prompt):
    model.eval()
    with torch.no_grad():
        template = "\n\nHuman: {user_content}\n\nAssistant: "
        prompt = template.format(user_content=prompt)
        inputs = tokenizer(
            prompt,
            padding=False,
            return_tensors='pt'
        ).to(device)
        generated_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=MAX_LENGTH
        )
        generated_text = tokenizer.decode(generated_ids[0])
        print(f"\nGenerated Text:\n{generated_text}\n")

    model.train()

def train(
    model_path: str,
    data_path: str,
    output_dir: str,
    prefix_length: int = 5,
    learning_rate: float = 1e-4,
    num_train_epochs: int = 3,
    batch_size: int = 2,
    project_name: str = "PrefixModelTraining",
    user_content: str = "Compose an engaging travel blog post about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions."
):
    wandb.init(project=project_name, config={
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "batch_size": batch_size,
        "prefix_length": prefix_length,
        "user_content": user_content
    })

    # Load the model and tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    model = PrefixModel(base_model, tokenizer=tokenizer)
    model.to(device)

    # trainable parameters
    trainable_params = [name for name, param in model.named_parameters() if param.requires_grad]
    print("Trainable parameters:", trainable_params)

    # Load and process data
    train_data = load_from_disk(data_path)
    process_fn = partial(process_ultrafeedback, tokenizer=tokenizer)
    train_data = train_data.map(process_fn, num_proc=8)

    print(train_data)

    # Set up training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=5,
        evaluation_strategy="no",
        save_strategy="epoch",
        report_to="wandb",
    )

    class CustomTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            outputs = model(**inputs)
            logits = outputs.logits
            batch_size, total_seq_length, vocab_size = logits.size()
            seq_length = inputs['input_ids'].size(1)
            prefix_length = total_seq_length - seq_length

            # Exclude prefix tokens from logits
            logits = logits[:, prefix_length:, :]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs['input_ids'][:, 1:].contiguous()

            shift_attention_mask = inputs['attention_mask'][:, 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)
            shift_attention_mask = shift_attention_mask.view(-1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            loss = loss_fct(shift_logits, shift_labels)

            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss = super().training_step(model, inputs)

            # Every 1000 steps generate a sample result
            if self.state.global_step % 500 == 0:
                print(f"\n[Step {self.state.global_step}] Generating text from the model:")
                generate_text(model, tokenizer, user_content)

            return loss

    custom_trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Start training
    custom_trainer.train()

    model.save_pretrained(output_dir)

    wandb.finish()

if __name__ == "__main__":
    fire.Fire(train)
