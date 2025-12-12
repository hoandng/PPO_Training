import sys
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    GenerationConfig
)
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

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


def train_ppo():
    print(">>> [3/4] Train PPO Strategy (Stable + Defensive Coding)...")

    # 1. SETUP GPU
    if torch.cuda.device_count() < 2:
        print("⚠️ Warning: Single GPU detected.")
        device_policy = 0
        device_reward = 0
    else:
        print("✅ Dual GPU detected.")
        device_policy = 0
        device_reward = 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # =================================================================
    # 2. LOAD POLICY MODEL (MANUAL + PATCHING) -> GPU 0
    # =================================================================
    print(f"Loading Policy Model on cuda:{device_policy}...")

    # A. Load Base Model
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device_policy},
        trust_remote_code=True
    )
    base_model = prepare_model_for_kbit_training(base_model)

    # B. Attach LoRA
    peft_config = LoraConfig(
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
        bias="none",
        lora_dropout=0.05
    )
    base_model = get_peft_model(base_model, peft_config)

    # C. Wrap TRL ValueHead
    model = AutoModelForCausalLMWithValueHead(base_model)
    model.v_head.requires_grad_(True)

    # =================================================================
    # 🛡️ MONKEY PATCHING BLOCK (KHẮC PHỤC MỌI LỖI ATTRIBUTE)
    # =================================================================
    print("🛡️ Applying Defensive Patches...")

    # Patch 1: "base_model_prefix" (Fix lỗi NoneType attribute)
    if not hasattr(model, "base_model_prefix"):
        model.base_model_prefix = "model"

    # Patch 2: "model" attribute (Fix lỗi object has no attribute 'model')
    # Một số version TRL gọi self.model, một số gọi self.pretrained_model
    if not hasattr(model, "model"):
        model.model = model.pretrained_model

    # Patch 3: "generation_config" (Fix lỗi generate)
    if not hasattr(model, "generation_config"):
        if hasattr(model.pretrained_model, "generation_config"):
            model.generation_config = model.pretrained_model.generation_config
        else:
            model.generation_config = GenerationConfig()

    # Patch 4: is_peft_model flag
    model.is_peft_model = True
    print("✅ Patches applied.")
    # =================================================================

    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # =================================================================
    # 3. LOAD REWARD MODEL -> GPU 1
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
    except Exception as e:
        print(f"❌ Error loading Reward Adapter: {e}")
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
    # 4. DATASET & TRAINER
    # =================================================================
    if not os.path.exists(Config.DATA_PPO_FILE):
        print("❌ Data PPO not found.")
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

    # Khởi tạo PPOTrainer (Với TRL 0.12.0 stable, dùng 'tokenizer', không dùng 'processing_class')
    ppo_trainer = PPOTrainer(
        args=config,
        model=model,
        ref_model=None,
        tokenizer=tokenizer,  # Quay lại dùng tokenizer (bản stable)
        train_dataset=dataset,
        data_collator=collator
    )

    # =================================================================
    # 5. TRAINING LOOP
    # =================================================================
    print(">>> Training PPO...")

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

        # A. Generate
        response_tensors = []
        for query in query_tensors:
            q_tensor = torch.tensor(query).to(f"cuda:{device_policy}").unsqueeze(0)
            r = ppo_trainer.generate(q_tensor, **generation_kwargs)
            response_tensors.append(r.squeeze()[len(query):])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # B. Reward
        texts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" for q, r in
                 zip(batch["query"], batch["response"])]
        pipe_outputs = reward_pipe(texts)

        rewards = []
        for output in pipe_outputs:
            score_good = 0.0
            for item in output:
                if str(item['label']) in ['1', 'LABEL_1']:
                    score_good = item['score']
                    break
            rewards.append(torch.tensor(score_good - 0.5).to(f"cuda:{device_policy}"))

        # C. Step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        step_count += 1
        print(f"Step {step_count}: Reward Avg = {torch.mean(torch.stack(rewards)).item():.4f}")

    # =================================================================
    # 6. SAVE
    # =================================================================
    print(">>> Saving...")
    if not os.path.exists(Config.PPO_ADAPTER_PATH):
        os.makedirs(Config.PPO_ADAPTER_PATH)

    # Save adapter của pretrained model
    model.pretrained_model.save_pretrained(Config.PPO_ADAPTER_PATH)
    print("✅ Done.")


if __name__ == "__main__":
    train_ppo()