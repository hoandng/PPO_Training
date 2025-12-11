import sys, os, torch
from datasets import load_dataset
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig, PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def train_ppo():
    print(">>> [3/4] Train PPO với Dual GPU Strategy...")

    # Kiểm tra GPU
    if torch.cuda.device_count() < 2:
        print("⚠️ Cảnh báo: Kaggle thường có 2 GPU T4. Đang chỉ tìm thấy 1. Có thể sẽ OOM.")
        device_policy = 0
        device_reward = 0
    else:
        print("✅ Đã phát hiện 2 GPU. Policy -> GPU 0, Reward -> GPU 1.")
        device_policy = 0
        device_reward = 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
    )

    # 1. Load Policy Model (Actor) -> GPU 0
    print(f"Loading Policy Model on cuda:{device_policy}...")
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        Config.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": device_policy},  # Ép vào GPU 0
        peft_config=LoraConfig(
            r=Config.LORA_R, lora_alpha=Config.LORA_ALPHA, task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"]
        )
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Reward Model -> GPU 1
    print(f"Loading Reward Model on cuda:{device_reward}...")
    rm_base = AutoModelForSequenceClassification.from_pretrained(
        Config.BASE_MODEL_NAME,
        num_labels=1,
        quantization_config=bnb_config,
        device_map={"": device_reward}  # Ép vào GPU 1
    )
    # Load Adapter vừa train ở bước trước
    rm_model = PeftModel.from_pretrained(rm_base, Config.REWARD_ADAPTER_PATH)

    # Tạo pipeline inference trên GPU 1
    reward_pipe = pipeline(
        "text-classification",
        model=rm_model,
        tokenizer=tokenizer,
        device=device_reward,
        batch_size=Config.BATCH_SIZE * 2
    )

    # 3. Prepare Dataset
    dataset = load_dataset("json", data_files=Config.DATA_PPO_FILE, split="train")

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"])
        return sample

    dataset = dataset.map(tokenize, batched=False)

    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # 4. Config PPO
    config = PPOConfig(
        model_name=Config.BASE_MODEL_NAME,
        learning_rate=Config.LEARNING_RATE,
        batch_size=Config.BATCH_SIZE,
        mini_batch_size=1,  # Giữ nhỏ để an toàn
        gradient_accumulation_steps=Config.GRAD_ACCUM_STEPS,
        log_with=None
    )

    ppo_trainer = PPOTrainer(config, model, tokenizer=tokenizer, dataset=dataset, data_collator=collator)

    # 5. Training Loop
    print(">>> Bắt đầu training...")
    generation_kwargs = {
        "min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id, "max_new_tokens": 128
    }

    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]

        # A. Generate (GPU 0)
        response_tensors = []
        for query in query_tensors:
            q_tensor = torch.tensor(query).to(f"cuda:{device_policy}").unsqueeze(0)
            r = ppo_trainer.generate(q_tensor, **generation_kwargs)
            response_tensors.append(r.squeeze()[len(query):])

        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # B. Score (GPU 1) -> Text đi qua CPU để chuyển GPU
        texts = [f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{r}<|im_end|>" for q, r in
                 zip(batch["query"], batch["response"])]

        # Inference trên GPU 1
        pipe_outputs = reward_pipe(texts, return_all_scores=True)
        rewards = [torch.tensor(output[0]["score"]).to(f"cuda:{device_policy}") for output in
                   pipe_outputs]  # Đưa reward về lại GPU 0

        # C. Update (GPU 0)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"Step {epoch}: Reward = {torch.mean(torch.stack(rewards)).item()}")

    # Save
    print(">>> Saving PPO Adapter...")
    ppo_trainer.save_pretrained(Config.PPO_ADAPTER_PATH)


if __name__ == "__main__":
    train_ppo()