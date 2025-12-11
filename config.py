import os
import sys
class Config:
    # Giá trị mặc định (Fallback) phòng khi quên set secret
    DEFAULT_BASE_MODEL = "khanhrill/HistoryGPT"
    DEFAULT_NEW_MODEL = "khanhrill/HistoryGPT-PPO-Kaggle"

    # --- HÀM LẤY SECRET AN TOÀN ---
    @staticmethod
    def get_secret(key, default_value):
        # 1. Thử lấy từ Kaggle Secrets
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            # get_secret sẽ raise lỗi nếu key không tồn tại, nên cần try/except
            return user_secrets.get_secret(key)
        except:
            pass

        # 2. Thử lấy từ Environment Variable (cho local/docker)
        env_val = os.getenv(key)
        if env_val:
            return env_val

        # 3. Trả về mặc định
        return default_value

    # --- LOAD CẤU HÌNH ---
    # Load Token & DB
    HF_TOKEN = get_secret.__func__("HF_TOKEN", None)
    DB_URI = get_secret.__func__("DB_URI", None)

    # Load Model Name từ Secret (Đây là phần bạn yêu cầu)
    BASE_MODEL_NAME = get_secret.__func__("BASE_MODEL", DEFAULT_BASE_MODEL)
    NEW_MODEL_NAME = get_secret.__func__("NEW_MODEL", DEFAULT_NEW_MODEL)

    # --- PATHS (Kaggle working dir) ---
    WORKING_DIR = "/kaggle/working"
    REWARD_ADAPTER_PATH = os.path.join(WORKING_DIR, "reward_adapter")
    PPO_ADAPTER_PATH = os.path.join(WORKING_DIR, "ppo_adapter")

    DATA_RM_FILE = os.path.join(WORKING_DIR, "data_rm.jsonl")
    DATA_PPO_FILE = os.path.join(WORKING_DIR, "data_ppo.jsonl")

    # --- HYPERPARAMETERS ---
    # Có thể đưa nốt Hyperparams vào Secret nếu muốn, nhưng để code cho gọn
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05

    LEARNING_RATE = 1.41e-5
    BATCH_SIZE = 2
    GRAD_ACCUM_STEPS = 4
    MAX_SEQ_LENGTH = 1024

    @classmethod
    def validate(cls):
        print(f">>> Cấu hình hiện tại:")
        print(f"    - Base Model: {cls.BASE_MODEL_NAME}")
        print(f"    - New Model Output: {cls.NEW_MODEL_NAME}")

        if not cls.HF_TOKEN:
            print("❌ Lỗi: Thiếu HF_TOKEN trong Secrets.")
            sys.exit(1)
        if not cls.DB_URI:
            print("⚠️ Cảnh báo: Thiếu DB_URI. Không thể truy xuất dữ liệu mới.")