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

# --- SETUP PATH ---
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

    # Chia train/test
    if len(dataset) > 20:
        dataset = dataset.train_test_split(test_size=0.1)
        train_ds = dataset["train"]
        eval_ds = dataset["test"]
    else:
        print("⚠️ Data quá ít, dùng toàn bộ để train.")
        train_ds = dataset
        eval_ds = None

    # 2. CONFIG 4-BIT
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    # 3. TOKENIZER
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Quan trọng: Set padding side là right cho Classification (khác với Gen)
    tokenizer.padding_side = "right"

    # 4. MODEL
    model = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    # 5. LORA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 6. PREPROCESS (FIX LỖI TENSOR)
    def preprocess(examples):
        # Ép padding="max_length" để mọi tensor có độ dài bằng nhau ngay lập tức
        # Điều này tốn VRAM hơn chút xíu nhưng sửa triệt để lỗi "Unable to create tensor"
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.MAX_SEQ_LENGTH,
            padding="max_length"
        )

    # QUAN TRỌNG: Thêm remove_columns=["text"]
    # Trainer sẽ báo lỗi nếu cột "text" (dạng string) vẫn còn tồn tại khi nó cố tạo Tensor.
    columns_to_remove = dataset["train"].column_names if "train" in dataset else dataset.column_names
    # Giữ lại 'label' nếu có, xóa 'text'
    if "label" in columns_to_remove:
        columns_to_remove.remove("label")
    if "input_ids" in columns_to_remove:  # Đề phòng
        columns_to_remove.remove("input_ids")

    print(f"Removing columns: {columns_to_remove}")

    tokenized_train = train_ds.map(preprocess, batched=True, remove_columns=columns_to_remove)
    tokenized_eval = eval_ds.map(preprocess, batched=True, remove_columns=columns_to_remove) if eval_ds else None

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {"accuracy": (predictions == labels).mean()}

    # 7. TRAINER
    training_args = TrainingArguments(
        output_dir=Config.REWARD_ADAPTER_PATH,
        per_device_train_batch_size=Config.BATCH_SIZE,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        num_train_epochs=Config.NUM_EPOCHS,
        learning_rate=2e-5,
        eval_strategy="steps" if eval_ds else "no",
        eval_steps=50,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        # Không cần DataCollatorWithPadding nữa vì ta đã padding="max_length" ở trên
        # Nhưng để mặc định cũng không sao
        compute_metrics=compute_metrics if eval_ds else None
    )

    trainer.train()

    print(">>> Saving Reward Model...")
    trainer.model.save_pretrained(Config.REWARD_ADAPTER_PATH)
    tokenizer.save_pretrained(Config.REWARD_ADAPTER_PATH)
    print("✅ Reward Model Saved.")


if __name__ == "__main__":
    train_reward()