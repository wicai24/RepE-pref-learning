import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk
import torch.nn as nn
import os
import numpy as np

# Constants
MAX_LENGTH = 768
PREFIX_LENGTH = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# PrefixModel class
class PrefixModel(nn.Module):
    def __init__(self, base_model, prefix_length=PREFIX_LENGTH):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_layer = nn.Linear(
            base_model.config.hidden_size, prefix_length * base_model.config.hidden_size
        ).to(torch.bfloat16)
        for param in self.base_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask, **kwargs):
        batch_size = input_ids.shape[0]
        input_embeds = self.base_model.get_input_embeddings()(input_ids)
        prefix_embeds = self.prefix_layer(input_embeds.mean(dim=1))
        prefix_embeds = prefix_embeds.view(batch_size, self.prefix_length, -1)
        prefixed_embeds = torch.cat([prefix_embeds, input_embeds], dim=1)
        prefix_attention_mask = torch.ones(
            batch_size, self.prefix_length, device=attention_mask.device
        )
        full_attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)
        outputs = self.base_model(
            inputs_embeds=prefixed_embeds, attention_mask=full_attention_mask, **kwargs
        )
        return outputs

# Load the base model
model_path = "/data/will_cai/wicai24/RepE-pref-learning/model/SFT/final_checkpoint"
base_model = AutoModelForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.bfloat16
).to(device)

# Load prefix layers
model_pos = PrefixModel(base_model).to(device)
model_neg = PrefixModel(base_model).to(device)

pos_prefix_layer_path = "/data/will_cai/wicai24/RepE-pref-learning/model/DPO/input_tuning_prefix_cosine_MSE_10_traj/prefix_model_pos_epoch_1.pt"
neg_prefix_layer_path = "/data/will_cai/wicai24/RepE-pref-learning/model/DPO/input_tuning_prefix_cosine_MSE_10_traj/prefix_model_neg_epoch_1.pt"

model_pos.prefix_layer.load_state_dict(torch.load(pos_prefix_layer_path))
model_neg.prefix_layer.load_state_dict(torch.load(neg_prefix_layer_path))

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token_id = tokenizer.eos_token_id

data_path = "../data/ultrafeedback/rm"
ultrafeedback_dataset = load_from_disk(data_path)

# Select the top 300 samples
top_300_data = ultrafeedback_dataset.select(range(300))

def process_data_point(data_point, tokenizer):
    template = "\n\nHuman: {user_content}\n\nAssistant: {assistant_content}"
    user_content = data_point["chosen"][0]["content"]
    assistant_content_chosen = data_point["chosen"][1]["content"]

    full_text_chosen = template.format(
        user_content=user_content, assistant_content=assistant_content_chosen
    )
    tokens = tokenizer(
        full_text_chosen,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

    input_ids = tokens["input_ids"].squeeze(0).to(device)
    attention_mask = tokens["attention_mask"].squeeze(0).to(device)
    return input_ids, attention_mask

# Function to get prefix embeddings
def get_prefix_embeddings(model, input_ids):
    input_embeds = model.base_model.get_input_embeddings()(input_ids)
    mean_embeds = input_embeds.mean(dim=1)
    prefix_embeds = model.prefix_layer(mean_embeds)
    prefix_embeds = prefix_embeds.view(
        -1, model.prefix_length, model.base_model.config.hidden_size
    )
    return prefix_embeds

# Function to map embeddings to tokens
def embeddings_to_tokens(prefix_embeds, model, tokenizer, top_k=1):
    token_embeddings = model.base_model.get_input_embeddings().weight
    token_embeddings = token_embeddings.to(prefix_embeds.device)

    tokens_list = []
    for idx in range(prefix_embeds.size(1)):
        prefix_vector = prefix_embeds[0, idx, :]
        similarities = torch.nn.functional.cosine_similarity(
            prefix_vector.unsqueeze(0), token_embeddings, dim=1
        )
        top_k_indices = torch.topk(similarities, k=top_k).indices
        tokens = tokenizer.convert_ids_to_tokens(top_k_indices.cpu().tolist())
        tokens_list.append(tokens)
    return tokens_list

# Prepare lists to collect tokens
pos_tokens_all = []
neg_tokens_all = []

for idx, data_point in enumerate(top_300_data):
    input_ids, attention_mask = process_data_point(data_point, tokenizer)
    input_ids = input_ids.unsqueeze(0)
    attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        # Positive model
        prefix_embeds_pos = get_prefix_embeddings(model_pos, input_ids)
        tokens_pos = embeddings_to_tokens(prefix_embeds_pos, model_pos, tokenizer)
        tokens_pos_flat = [token[0] for token in tokens_pos]
        pos_tokens_all.append(tokens_pos_flat)

        # Negative model
        prefix_embeds_neg = get_prefix_embeddings(model_neg, input_ids)
        tokens_neg = embeddings_to_tokens(prefix_embeds_neg, model_neg, tokenizer)
        tokens_neg_flat = [token[0] for token in tokens_neg]
        neg_tokens_all.append(tokens_neg_flat)

# Save all tokens into one file
output_dir = "prefix_outputs"
os.makedirs(output_dir, exist_ok=True)

output_tokens_pos_path = os.path.join(output_dir, "all_prefix_pos_tokens.txt")
with open(output_tokens_pos_path, "w", encoding="utf-8") as f:
    for tokens in pos_tokens_all:
        f.write(" ".join(tokens) + "\n")

output_tokens_neg_path = os.path.join(output_dir, "all_prefix_neg_tokens.txt")
with open(output_tokens_neg_path, "w", encoding="utf-8") as f:
    for tokens in neg_tokens_all:
        f.write(" ".join(tokens) + "\n")
