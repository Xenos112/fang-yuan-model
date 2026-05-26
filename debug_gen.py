from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import Paths, ModelConfig

model_path = str(Paths["model_output"])
if not Paths["model_output"].exists():
    model_path = ModelConfig["model_name"]

print(f"Loading {model_path}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=512,
    load_in_4bit=ModelConfig["load_in_4bit"],
    dtype=None,
    device_map="auto",
)
tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")
FastLanguageModel.for_inference(model)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Analyze the situation and respond as Fang Yuan.\nContext: A young man asks you who you are."
messages = [
    {"role": "user", "content": prompt},
]
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_tensors="pt",
    enable_thinking=True,
).to("cuda")

print(f"Input shape: {inputs.shape}")
print(f"Input tokens: {inputs[0][:20]}")

attention_mask = (inputs != tokenizer.pad_token_id).long().to("cuda")

print("Generating...")
import time
start = time.time()

outputs = model.generate(
    inputs,
    attention_mask=attention_mask,
    max_new_tokens=50,
    temperature=0.7,
    do_sample=True,
    use_cache=True,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.pad_token_id,
)

elapsed = time.time() - start
print(f"Generated {outputs.shape[1] - inputs.shape[1]} tokens in {elapsed:.1f}s")

response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"\nFang Yuan: {response}")
