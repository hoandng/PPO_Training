import sys, os, torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, \
    Trainer, DataCollatorWithPadding
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def train_reward():
    print(">>> [2/4] Train Reward Model (Binary Classification)...")

    if not os.path.exists(Config.DATA_RM_FILE) or os.path.getsize(Config.DATA_RM_FILE) == 0:
        print("❌ Lỗi: File data RM rỗng.")
        return

    # Load dataset
    dataset = load_dataset("json", data_files=Config.DATA_RM_FILE, split="train")

    # Chia train/test để đánh giá accuracy
    dataset = dataset.train_test_split(test_size=0.1)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        Config.BASE_MODEL_NAME,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Load Model Classification (2 nhãn: 0=Bad, 1=Good)
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)

    # Hàm Tokenize
    def preprocess(examples):
        return tokenizer(examples["text"], truncation=True, max_length=Config.MAX_SEQ_LENGTH)

    tokenized_datasets = dataset.map(preprocess, batched=True)

    # Định nghĩa Metric để xem độ chính xác
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    # Trainer Config
    training_args = TrainingArguments(
        output_dir=Config.REWARD_ADAPTER_PATH,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,  # Có thể tăng lên 2-3 nếu data ít
        learning_rate=2e-5,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",  # Không save checkpoint rác
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.model.save_pretrained(Config.REWARD_ADAPTER_PATH)
    print("✅ Reward Model (Classifier) Saved.")


if __name__ == "__main__":
    train_reward()