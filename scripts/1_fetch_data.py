import sys, os, json
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path: sys.path.insert(0, parent_dir)
from config import Config


def fetch_data():
    print(">>> [1/4] Đang lấy dữ liệu Point-wise (Like/Dislike)...")

    if not Config.DB_URI:
        print("❌ Lỗi: Không có DB_URI.")
        return

    try:
        engine = create_engine(Config.DB_URI)

        # Query lấy dữ liệu phẳng, không cần join phức tạp nếu bảng chat đã chứa đủ
        # Dựa trên ảnh của bạn, tôi điều chỉnh query để lấy rating 1 và -1
        query = """
                select t.chat::json->'history'->'messages'->(message.value::json->>'parentId')->>'content' as question,
            message.value::json->>'content' AS answer,
            message.value::json->'annotation'->>'rating' as rating
                from chat t cross join lateral json_each(t.chat::json#>'{history, messages}') as message
                    inner join public.user u \
                on t.user_id = u.id
                where message.value::json->'annotation' is not null \
                """

        df = pd.read_sql(query, engine)
        print(f"📊 Tìm thấy {len(df)} dòng feedback.")

        rm_data = []  # Dữ liệu cho Reward Model (Input + Label)
        ppo_data = []  # Dữ liệu cho PPO (Chỉ cần Question)

        for index, row in df.iterrows():
            try:
                question = row['question']
                answer = row['answer']
                rating = int(row['rating'])  # 1 hoặc -1

                # Logic gán nhãn: 1 -> Label 1 (Good), -1 -> Label 0 (Bad)
                if rating == 1:
                    label = 1
                elif rating == -1:
                    label = 0
                else:
                    continue  # Bỏ qua nếu rating bằng 0 hoặc null

                # Thêm vào dataset train Reward Model
                # Text format: User: ... \n Assistant: ...
                full_text = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>"

                rm_data.append({
                    "text": full_text,
                    "label": label
                })

                # Thêm vào dataset PPO (Chỉ cần câu hỏi để model tự sinh câu trả lời mới)
                ppo_data.append({"query": question})

            except Exception as e:
                continue

        # Fallback nếu không có data (để test pipeline)
        if len(rm_data) == 0:
            print("⚠️ Cảnh báo: Không có dữ liệu thật. Tạo Dummy Data.")
            rm_data = [
                          {"text": "User: Hi\nAssistant: Hello (Good)", "label": 1},
                          {"text": "User: Hi\nAssistant: ... (Bad)", "label": 0}
                      ] * 10
            ppo_data = [{"query": "Hi"}] * 10

        # Lưu file
        pd.DataFrame(rm_data).to_json(Config.DATA_RM_FILE, orient="records", lines=True)
        pd.DataFrame(ppo_data).to_json(Config.DATA_PPO_FILE, orient="records", lines=True)
        print(f"✅ Đã lưu: {len(rm_data)} mẫu RM (Classification), {len(ppo_data)} prompt PPO.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")


if __name__ == "__main__":
    fetch_data()