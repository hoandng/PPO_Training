import sys
import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel

# --- SETUP PATH ĐỂ IMPORT CONFIG ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from config import Config
except ImportError:
    # Fallback nếu chạy trực tiếp trong thư mục scripts
    sys.path.append("/kaggle/working/HistoryGPT-Kaggle-PPO")
    from config import Config


def train_ppo():
    print(">>> [3/4] Train PPO với Dual GPU Strategy (Classification Reward)...")

    # 1. Cấu hình GPU (Kaggle T4 x2)
    if torch.cuda.device_count() < 2:
        print("⚠️ Cảnh báo: Chỉ tìm thấy 1 GPU. Có thể gặp lỗi OOM.")
        device_policy = 0
        device_reward = 0
    else:
        print("✅ Đã phát hiện 2 GPU. Policy -> GPU 0, Reward -> GPU 1.")
        device_policy = 0
        device_reward = 1

    # Config Quantization (4-bit) để tiết kiệm VRAM
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # =================================================================
    # 2. LOAD POLICY MODEL (ACTOR) -> GPU 0
    # =================================================================
    print(f"Loading Policy Model on cuda:{device_policy}...")

    # PPO cần model có Value Head
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        Config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device_policy},  # Ép vào GPU 0
        trust_remote_code=True,
        peft_config=LoraConfig(
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(
        Config.BASE_MODEL_NAME,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =================================================================
    # 3. LOAD REWARD MODEL (JUDGE) -> GPU 1
    # =================================================================
    print(f"Loading Reward Model on cuda:{device_reward}...")

    # Load Base Model (Classification - 2 labels)
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,  # Quan trọng: 2 nhãn (0: Bad, 1: Good)
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map={"": device_reward}  # Ép vào GPU 1
    )

    # Load Adapter đã train ở bước 2
    try:
        rm_model = PeftModel.from_pretrained(rm_base, Config.REWARD_ADAPTER_PATH)
        print("✅ Đã load Reward Adapter thành công.")
    except Exception as e:
        print(f"❌ Lỗi load Reward Adapter: {e}")
        return

    # Tạo pipeline inference trên GPU 1
    # top_k=None để trả về score của cả 2 nhãn (Good/Bad)
    reward_pipe = pipeline(
        "text-classification",
        model=rm_model,
        tokenizer=tokenizer,
        batch_size=Config.BATCH_SIZE * 2,
        top_k=None
    )

    # =================================================================
    # 4. CHUẨN BỊ DATASET & PPO CONFIG
    # =================================================================
    if not os.path.exists(Config.DATA_PPO_FILE):
        print("❌ Không tìm thấy file data PPO.")
        return

    dataset = load_dataset("json", data_files=Config.DATA_PPO_FILE, split="train")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    config = PPOConfig(
        learning_rate=Config.LEARNING_RATE,
        batch_size=Config.BATCH_SIZE,
        mini_batch_size=1,  # Giữ nhỏ để an toàn
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        target_kl=0.1  # Giữ model không đi quá xa model gốc
    )

    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        data_collator=collator
    )

    # =================================================================
    # 5. TRAINING LOOP
    # =================================================================
    print(">>> Bắt đầu training PPO...")

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 0.9,  # Dùng top_p sampling cho tự nhiên
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    step_count = 0

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # --- A. GENERATE (POLICY MODEL - GPU 0) ---
        response_tensors = []
        for query in query_tensors:
            # Chuyển query sang GPU 0
            q_tensor = torch.tensor(query).to(f"cuda:{device_policy}").unsqueeze(0)

            # Generate câu trả lời
            r = ppo_trainer.generate(q_tensor, **generation_kwargs)

            # Cắt bỏ phần query, chỉ giữ lại phần answer mới sinh ra
            response_tensors.append(r.squeeze()[len(query):])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # --- B. COMPUTE REWARD (REWARD MODEL - GPU 1) ---
        # Tạo prompt đầy đủ cho Reward Model chấm
        texts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" for q, r in
                 zip(batch["query"], batch["response"])]

        # Inference trên GPU 1
        pipe_outputs = reward_pipe(texts)

        rewards = []
        for output in pipe_outputs:
            # output dạng: [{'label': 'LABEL_0', 'score': 0.1}, {'label': 'LABEL_1', 'score': 0.9}]
            # LABEL_1 là lớp "Tốt" (Rating = 1)
            score_good = 0.0
            for item in output:
                # Kiểm tra label nào là label Positive
                if item['label'] == 'LABEL_1' or item['label'] == '1':
                    score_good = item['score']
                    break

            # Tính Reward:
            # score_good chạy từ 0 đến 1.
            # Trừ đi 0.5 để reward chạy từ -0.5 (Rất tệ) đến +0.5 (Rất tốt) -> Giúp PPO hội tụ nhanh hơn
            final_reward = score_good - 0.5

            # Chuyển reward về GPU 0 để PPO Trainer dùng
            rewards.append(torch.tensor(final_reward).to(f"cuda:{device_policy}"))

        # --- C. UPDATE POLICY (GPU 0) ---
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        step_count += 1
        avg_reward = torch.mean(torch.stack(rewards)).item()
        print(f"Step {step_count}: Reward Trung bình = {avg_reward:.4f}")

    # =================================================================
    # 6. SAVE ADAPTER
    # =================================================================
    print(">>> Saving PPO Adapter...")
    if not os.path.exists(Config.PPO_ADAPTER_PATH):
        os.makedirs(Config.PPO_ADAPTER_PATH)

    ppo_trainer.save_pretrained(Config.PPO_ADAPTER_PATH)
    print(f"✅ Đã lưu Adapter tại: {Config.PPO_ADAPTER_PATH}")


if __name__ == "__main__":
    # Validate Config trước khi chạy
    try:
        Config.validate()
    except:
        pass  # Bỏ qua nếu chạy test cục bộ

    train_ppo()