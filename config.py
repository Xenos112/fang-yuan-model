from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

Paths = {
    "epub": BASE_DIR / "data" / "reverend-insanity.epub",
    "dataset": BASE_DIR / "data" / "generated.json",
    "model_output": BASE_DIR / "models" / "fang-yuan-gemma",
}

# Hyperparameters for Unsloth
ModelConfig = {
    "model_name": "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    "max_seq_length": 2048,
    "load_in_4bit": True,
}
