import argparse
import sys
from unsloth import FastLanguageModel
from config import Paths, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--base", action="store_true")
    args = parser.parse_args()

    if args.base:
        model_name = ModelConfig["model_name"]
    elif args.model:
        model_name = args.model
    else:
        trained_path = Paths["model_output"]
        if trained_path.exists():
            model_name = str(trained_path)
        else:
            model_name = ModelConfig["model_name"]
            print(f"No trained model found, using base model {model_name}")

    print(f"Loading {model_name}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=ModelConfig["max_seq_length"],
        load_in_4bit=ModelConfig["load_in_4bit"],
        dtype=None,
        device_map="auto",
    )
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n=== Fang Yuan Chat ===")
    print("Describe a situation — he'll respond in-character.")
    print("Type 'quit' to exit.\n")

    while True:
        context = input("You: ").strip()
        if context.lower() in ("quit", "exit", ""):
            break

        messages = [
            {"role": "user", "content": [{"type": "text", "text": f"Analyze the situation and respond as Fang Yuan.\nContext: {context}"}]},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=True,
        ).to("cuda")

        attention_mask = (inputs != tokenizer.pad_token_id).long().to("cuda")

        print("Fang Yuan: ", end="", flush=True)

        from transformers import TextStreamer
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )
        print()


if __name__ == "__main__":
    main()
