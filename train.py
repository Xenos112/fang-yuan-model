import json
import torch
from datasets import Dataset, DatasetDict
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
from config import Paths, ModelConfig


def load_dataset_from_json(path: str) -> DatasetDict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted = []
    for entry in data:
        messages = [
            {"role": "user", "content": f"{entry['instruction']}\n{entry['input']}"},
            {"role": "assistant", "content": entry["output"]},
        ]
        formatted.append({"messages": messages})

    full = Dataset.from_list(formatted)
    split = full.train_test_split(test_size=0.05, seed=42)
    return DatasetDict({"train": split["train"], "test": split["test"]})


def main():
    dataset = load_dataset_from_json(Paths["dataset"])
    print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ModelConfig["model_name"],
        max_seq_length=ModelConfig["max_seq_length"],
        load_in_4bit=ModelConfig["load_in_4bit"],
        dtype=None,
        device_map="auto",
    )

    tokenizer = get_chat_template(tokenizer, chat_template="gemma")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    sft_config = SFTConfig(
        output_dir=str(Paths["model_output"]),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        learning_rate=2e-4,
        embedding_learning_rate=2e-5,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        dataset_text_field="",
        max_seq_length=ModelConfig["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {Paths['model_output']}...")
    model.save_pretrained_merged(
        str(Paths["model_output"]),
        tokenizer,
        save_method="merged_16bit",
    )
    print("Done!")


if __name__ == "__main__":
    main()
