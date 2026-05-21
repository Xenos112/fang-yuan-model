"""Convert the trained model to GGUF for instant loading."""
from unsloth import FastLanguageModel
from config import Paths, ModelConfig

model_path = str(Paths["model_output"])
gguf_path = str(Paths["model_output"] / "gguf")

print(f"Loading trained model from {model_path} (slow once)...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=ModelConfig["max_seq_length"],
    load_in_4bit=True,
    dtype=None,
    device_map="auto",
)

print(f"Saving as GGUF Q4_K_M to {gguf_path}...")
model.save_pretrained_gguf(
    gguf_path,
    tokenizer,
    quantization_method="q4_k_m",
)
print("Done! GGUF saved.")
