"""One-time conversion of the 16-bit merged model to 4-bit for fast loading."""
import torch
from unsloth import FastLanguageModel
from config import Paths, ModelConfig

model_path = str(Paths["model_output"])
output_path = str(Paths["model_output"] / "4bit")

print(f"Loading 16-bit merged model from {model_path} (this will be slow once)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=ModelConfig["max_seq_length"],
    load_in_4bit=True,
    dtype=None,
    device_map="auto",
)

print(f"Saving as 4-bit to {output_path}...")
model.save_pretrained_merged(
    output_path,
    tokenizer,
    save_method="merged_4bit_forced",
)
print("Done! Future loads will be fast.")
