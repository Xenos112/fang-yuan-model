import argparse
from unsloth import FastLanguageModel
from config import Paths, ModelConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default=None,
                        help="Model path or name. Default: trained model if exists, else base model.")
    parser.add_argument("--base", action="store_true",
                        help="Use the base HuggingFace model instead of trained model.")
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
            print(f"No trained model found at {trained_path}, using base model {model_name}")

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

    print("\nFang Yuan model ready. Type 'quit' to exit.\n")

    while True:
        context = input("Context (situation): ").strip()
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
        ).to("cuda")

        attention_mask = (inputs != tokenizer.pad_token_id).long().to("cuda")

        outputs = model.generate(
            inputs,
            attention_mask=attention_mask,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            use_cache=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\nFang Yuan: {response}\n")


if __name__ == "__main__":
    main()
