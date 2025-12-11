import sys, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def merge_push():
    print(">>> [4/4] Merge & Push...")
    login(token=Config.HF_TOKEN)

    # Load Base (CPU để tiết kiệm VRAM khi merge)
    base_model = AutoModelForCausalLM.from_pretrained(
        Config.BASE_MODEL_NAME, return_dict=True, torch_dtype=torch.float16, device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(Config.BASE_MODEL_NAME)

    # Load PPO Adapter
    print("Loading Adapter...")
    model = PeftModel.from_pretrained(base_model, Config.PPO_ADAPTER_PATH)
    model = model.merge_and_unload()

    print(f"Pushing to: {Config.NEW_MODEL_NAME}")
    model.push_to_hub(Config.NEW_MODEL_NAME, private=True)
    tokenizer.push_to_hub(Config.NEW_MODEL_NAME, private=True)
    print("✅ Hoàn tất Pipeline!")


if __name__ == "__main__":
    Config.validate()
    merge_push()