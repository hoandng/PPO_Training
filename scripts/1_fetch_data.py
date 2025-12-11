import sys, os, json
import pandas as pd
from sqlalchemy import create_engine

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def fetch_data():
    print(">>> [1/4] Đang lấy dữ liệu từ Database...")

    if not Config.DB_URI:
        print("⚠️ Cảnh báo: Không có DB_URI. Kiểm tra lại Secrets.")
        return

    try:
        engine = create_engine(Config.DB_URI)
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

        rm_data = []  # Dữ liệu cặp (Chosen/Rejected)
        ppo_data = []  # Dữ liệu prompt (Question only)

        grouped = df.groupby('question')
        for question, group in grouped:
            ppo_data.append({"query": question})

            try:
                group['rating'] = group['rating'].astype(float)
                sorted_group = group.sort_values('rating', ascending=False)

                if len(sorted_group) >= 2:
                    best = sorted_group.iloc[0]
                    worst = sorted_group.iloc[-1]
                    # Logic: Rating lệch nhau thì mới học
                    if best['rating'] > worst['rating']:
                        rm_data.append({
                            "prompt": question,
                            "chosen": best['answer'],
                            "rejected": worst['answer']
                        })
            except:
                continue

        # Lưu file
        pd.DataFrame(rm_data).to_json(Config.DATA_RM_FILE, orient="records", lines=True)
        pd.DataFrame(ppo_data).to_json(Config.DATA_PPO_FILE, orient="records", lines=True)
        print(f"✅ Đã lưu: {len(rm_data)} cặp RM, {len(ppo_data)} prompt PPO.")

    except Exception as e:
        print(f"❌ Lỗi SQL: {e}")


if __name__ == "__main__":
    Config.validate()
    fetch_data()