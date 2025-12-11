import sys
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

# --- SETUP PATH ĐỂ IMPORT CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import Config
except ImportError:
    sys.path.append("/kaggle/working/HistoryGPT-Kaggle-PPO")
    from config import Config


def train_reward():
    print(">>> [2/4] Train Reward Model (Binary Classification)...")

    # Check data
    if not os.path.exists(Config.DATA_RM_FILE) or os.path.getsize(Config.DATA_RM_FILE) == 0:
        print("❌ Lỗi: File data RM rỗng hoặc không tồn tại.")
        return

    # 1. LOAD DATASET
    dataset = load_dataset("json", data_files=Config.DATA_RM_FILE, split="train")

    # Chia train/test an toàn (tránh lỗi nếu data < 10 dòng)
    if len(dataset) > 20:
        dataset = dataset.train_test_split(test_size=0.1)
        train_ds = dataset["train"]
        eval_ds = dataset["test"]
    else:
        print("⚠️ Data quá ít, dùng toàn bộ để train (không eval).")
        train_ds = dataset
        eval_ds = None

    # 2. CONFIG QUANTIZATION (4-BIT)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. LOAD TOKENIZER
    # Thêm trust_remote_code=True cho Qwen
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. LOAD MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True  # Quan trọng với Qwen
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Bật gradient checkpointing để tiết kiệm VRAM (Quan trọng trên Kaggle)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 5. CONFIG LORA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"]  # Target module chuẩn của Qwen
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. PREPROCESS DATA
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            padding=False  # Để DataCollator tự xử lý padding dynamic sẽ nhanh hơn
        )

    tokenized_train = train_ds.map(preprocess, batched=True)
    tokenized_eval = eval_ds.map(preprocess, batched=True) if eval_ds else None

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    # 7. TRAINING ARGUMENTS (CẬP NHẬT CÁC THAM SỐ MỚI)
    training_args = TrainingArguments(
        output_dir=Config.REWARD_ADAPTER_PATH,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        learning_rate=2e-5,

        # --- CÁC THAM SỐ QUAN TRỌNG ĐÃ KIỂM TRA ---
        eval_strategy="steps" if eval_ds else "no",  # Dùng 'eval_strategy' thay vì 'evaluation_strategy'
        eval_steps=50,
        logging_steps=10,
        save_strategy="no",  # Không lưu checkpoint rác tốn dung lượng

        fp16=True,  # T4 dùng fp16 tốt hơn bf16
        gradient_checkpointing=True,  # Bắt buộc bật để tránh OOM
        remove_unused_columns=False,  # Tránh lỗi Trainer tự xóa cột data
        report_to="none"
    )

    # 8. TRAINER
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        # tokenizer=tokenizer, <--- ĐÃ XÓA (Gây lỗi ở bản mới)
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics if eval_ds else None
    )

    # Bắt đầu train
    trainer.train()

    # Save Model & Tokenizer
    print(">>> Saving Reward Model...")
    trainer.model.save_pretrained(Config.REWARD_ADAPTER_PATH)
    tokenizer.save_pretrained(Config.REWARD_ADAPTER_PATH)
    print("✅ Reward Model (Classifier) Saved successfully.")


if __name__ == "__main__":
    # Validate Config
    try:
        Config.validate()
    except:
        pass
    train_reward()