import sys, os, torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, TaskType
from trl import RewardTrainer, RewardConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def train_reward():
    print(">>> [2/4] Train Reward Model...")

    if not os.path.exists(Config.DATA_RM_FILE):
        print("Không tìm thấy data RM. Bỏ qua bước này.")
        return

    dataset = load_dataset("json", data_files=Config.DATA_RM_FILE, split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME, num_labels=1, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT, target_modules=["q_proj", "v_proj"]
    )

    def preprocess(examples):
        new_examples = {"input_ids_chosen": [], "attention_mask_chosen": [], "input_ids_rejected": [],
                        "attention_mask_rejected": []}
        for prompt, chosen, rejected in zip(examples["prompt"], examples["chosen"], examples["rejected"]):
            # Format đơn giản hoặc dùng template
            c_txt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen}<|im_end|>"
            r_txt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{rejected}<|im_end|>"

            tok_c = tokenizer(c_txt, truncation=True, max_length=Config.MAX_SEQ_LENGTH)
            tok_r = tokenizer(r_txt, truncation=True, max_length=Config.MAX_SEQ_LENGTH)

            new_examples["input_ids_chosen"].append(tok_c["input_ids"])
            new_examples["attention_mask_chosen"].append(tok_c["attention_mask"])
            new_examples["input_ids_rejected"].append(tok_r["input_ids"])
            new_examples["attention_mask_rejected"].append(tok_r["attention_mask"])
        return new_examples

    dataset = dataset.map(preprocess, batched=True)

    trainer = RewardTrainer(
        model=model, tokenizer=tokenizer, peft_config=peft_config, train_dataset=dataset,
        args=RewardConfig(
            output_dir=Config.REWARD_ADAPTER_PATH,
            per_device_train_batch_size=Config.BATCH_SIZE,
            gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
            num_train_epochs=1,
            report_to="none",
            fp16=True
        )
    )
    trainer.train()
    trainer.model.save_pretrained(Config.REWARD_ADAPTER_PATH)
    print("✅ Reward Model Adapter Saved.")


if __name__ == "__main__":
    train_reward()