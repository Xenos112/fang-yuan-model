from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import ModelConfig


def main():
    model_path = input("Model path (enter for default): ").strip()
    if not model_path:
        from config import Paths
        model_path = str(Paths["model_output"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=ModelConfig["max_seq_length"],
        load_in_4bit=ModelConfig["load_in_4bit"],
        dtype=None,
        device_map="auto",
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma")
    FastLanguageModel.for_inference(model)

    print("\nFang Yuan model ready. Type 'quit' to exit.\n")

    while True:
        context = input("Context (situation): ").strip()
        if context.lower() in ("quit", "exit"):
            break

        messages = [
            {"role": "user", "content": f"Analyze the situation and respond as Fang Yuan.\nContext: {context}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        print(f"\nFang Yuan: {response}\n")


if __name__ == "__main__":
    main()
