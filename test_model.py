import json
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from config import Paths, ModelConfig


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(Paths["model_output"]),
        max_seq_length=ModelConfig["max_seq_length"],
        load_in_4bit=ModelConfig["load_in_4bit"],
        dtype=None,
        device_map="auto",
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4-thinking")
    FastLanguageModel.for_inference(model)

    with open(Paths["dataset"], "r", encoding="utf-8") as f:
        data = json.load(f)

    test_samples = data[::20][:5]

    for i, entry in enumerate(test_samples):
        messages = [
            {"role": "user", "content": f"{entry['instruction']}\n{entry['input']}"},
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
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

        print(f"=== Sample {i+1} ===")
        print(f"Context: {entry['input']}")
        print(f"Expected: {entry['output']}")
        print(f"Model:    {response}\n")


if __name__ == "__main__":
    main()
