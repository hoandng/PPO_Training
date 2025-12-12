import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

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


def train_ppo():
    print(">>> [3/4] Train PPO với Dual GPU Strategy (Manual Loading)...")

    # 1. Cấu hình GPU
    if torch.cuda.device_count() < 2:
        print("⚠️ Cảnh báo: Chỉ tìm thấy 1 GPU. Có thể gặp lỗi OOM.")
        device_policy = 0
        device_reward = 0
    else:
        print("✅ Đã phát hiện 2 GPU. Policy -> GPU 0, Reward -> GPU 1.")
        device_policy = 0
        device_reward = 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # =================================================================
    # 2. LOAD POLICY MODEL (THỦ CÔNG TỪNG BƯỚC) -> GPU 0
    # =================================================================
    print(f"Loading Policy Model on cuda:{device_policy}...")

    # BƯỚC A: Load Base Model bằng Transformers thuần (Tránh lỗi TRL)
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device_policy},
        trust_remote_code=True
    )

    # BƯỚC B: Chuẩn bị model cho training 4-bit (QUAN TRỌNG)
    base_model = prepare_model_for_kbit_training(base_model)

    # BƯỚC C: Gắn LoRA thủ công
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05
    )
    base_model = get_peft_model(base_model, peft_config)

    # BƯỚC D: Đóng gói vào TRL Wrapper (Value Head)
    # Thay vì dùng .from_pretrained (gây lỗi), ta khởi tạo trực tiếp từ base_model
    model = AutoModelForCausalLMWithValueHead(base_model)

    # Kích hoạt gradient cho Value Head (vì nó là lớp mới thêm vào)
    model.v_head.requires_grad_(True)

    # BƯỚC E: Vá lỗi generation_config (cho Qwen)
    if hasattr(model.pretrained_model, "generation_config"):
        model.generation_config = model.pretrained_model.generation_config
    else:
        # Fallback tạo config mặc định
        from transformers import GenerationConfig
        try:
            model.generation_config = GenerationConfig.from_pretrained(Config.BASE_MODEL_NAME, trust_remote_code=True)
        except:
            model.generation_config = GenerationConfig()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =================================================================
    # 3. LOAD REWARD MODEL (JUDGE) -> GPU 1
    # =================================================================
    print(f"Loading Reward Model on cuda:{device_reward}...")

    rm_base = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map={"": device_reward},
        trust_remote_code=True
    )

    try:
        rm_model = PeftModel.from_pretrained(rm_base, Config.REWARD_ADAPTER_PATH)
        print("✅ Đã load Reward Adapter thành công.")
    except Exception as e:
        print(f"❌ Lỗi load Reward Adapter: {e}")
        return

    # Pipeline
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
        mini_batch_size=1,
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
    )

    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        ref_model=None,
        reward_model=None,
        value_model=None,
        processing_class=tokenizer,  # Đã sửa thành processing_class
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
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 128
    }

    step_count = 0

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # A. GENERATE (GPU 0)
        response_tensors = []
        for query in query_tensors:
            q_tensor = torch.tensor(query).to(f"cuda:{device_policy}").unsqueeze(0)
            r = ppo_trainer.generate(q_tensor, **generation_kwargs)
            response_tensors.append(r.squeeze()[len(query):])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # B. COMPUTE REWARD (GPU 1)
        texts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" for q, r in
                 zip(batch["query"], batch["response"])]

        # Inference
        pipe_outputs = reward_pipe(texts)

        rewards = []
        for output in pipe_outputs:
            score_good = 0.0
            for item in output:
                # Lấy score của nhãn '1' (Tốt)
                if str(item['label']) in ['1', 'LABEL_1']:
                    score_good = item['score']
                    break

            # Tính reward
            final_reward = score_good - 0.5
            rewards.append(torch.tensor(final_reward).to(f"cuda:{device_policy}"))

        # C. UPDATE (GPU 0)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        step_count += 1
        avg_reward = torch.mean(torch.stack(rewards)).item()
        print(f"Step {step_count}: Reward Avg = {avg_reward:.4f}")

    # =================================================================
    # 6. SAVE ADAPTER
    # =================================================================
    print(">>> Saving PPO Adapter...")
    if not os.path.exists(Config.PPO_ADAPTER_PATH):
        os.makedirs(Config.PPO_ADAPTER_PATH)

    # Lưu ý: Khi save thủ công kiểu này, ta cần save adapter của pretrained_model
    # model.pretrained_model chính là cái PeftModel ta tạo ở Bước C
    model.pretrained_model.save_pretrained(Config.PPO_ADAPTER_PATH)
    print(f"✅ Đã lưu Adapter tại: {Config.PPO_ADAPTER_PATH}")


if __name__ == "__main__":
    train_ppo()