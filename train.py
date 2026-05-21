import json
import pyarrow as pa
from datasets import Dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only
from trl import SFTTrainer, SFTConfig
from config import Paths, ModelConfig


def main():
    with open(Paths["dataset"], "r", encoding="utf-8") as f:
        data = json.load(f)

    # Load model first so we can use its tokenizer to format data
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=ModelConfig["model_name"],
        max_seq_length=ModelConfig["max_seq_length"],
        load_in_4bit=ModelConfig["load_in_4bit"],
        dtype=None,
        device_map="auto",
    )

    # The 'it' model already has the correct Gemma 4 chat template — use it
    texts = []
    for entry in data:
        messages = [
            {"role": "user", "content": f"{entry['instruction']}\n{entry['input']}"},
            {"role": "assistant", "content": entry["output"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(text)

    table = pa.table({"text": pa.array(texts, type=pa.string())})
    full = Dataset(table, fingerprint="00000")
    split = full.train_test_split(test_size=0.05, seed=42)

    print(f"Train: {len(split['train'])} | Test: {len(split['test'])}")
    print(f"Example:\n{texts[0]}\n")

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
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
        dataset_text_field="text",
        max_seq_length=ModelConfig["max_seq_length"],
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=sft_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn|>user\n",
        response_part="<|turn|>model\n",
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
