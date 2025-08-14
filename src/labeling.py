import os
import time
from dotenv import load_dotenv
import logging
import sqlite3
import pandas as pd
from openai import OpenAI

class BatchLabeler:
    def __init__(self, api_key: str, input_path: str, db_path: str = "labels.db", batch_size: int = 10, sleep_time: float = 1.5):
        self.client = OpenAI(api_key=api_key)
        self.input_path = input_path
        self.db_path = db_path
        self.batch_size = batch_size
        self.sleep_time = sleep_time

        self._setup_logger()
        self._init_db()
        self._check_api()

    def _setup_logger(self):
        logging.basicConfig(
            filename="labeling.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger("BatchLabeler")

    def _init_db(self):
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS labels (
                peerid INTEGER PRIMARY KEY,
                peer_name TEXT,
                about TEXT,
                label TEXT
            )
        """)
        self.conn.commit()

    def _check_api(self):
        try:
            test_response = self.client.models.list()
            print("✅ اتصال موفق به OpenAI API برقرار شد.")  # ← اضافه کن برای تست
            self.logger.info("✅ اتصال موفق به OpenAI API برقرار شد.")
        except Exception as e:
            print(f"❌ اتصال به OpenAI API ناموفق بود: {e}")
            self.logger.error(f"❌ اتصال به OpenAI API ناموفق بود: {e}")
            raise


    def get_topic_label(self, name: str, about: str, model: str = "gpt-4") -> str:
        prompt = f"""گروهی در تلگرام با این مشخصات داریم:

        نام گروه: {name}
        توضیحات: {about}

این گروه در چه زمینه هست در یک کلمه مشخص کن این گروه در چه دسته بندی قرار میگیرد . این ها گروهای تلگرامی هستن با توجه به این که این که تلگرام یک شبکه اجتماعی هست تو باید بگویی که این گروها در چه زمنیه هستند فقط دو کلمه به عنوان لیبل بگو 
        """
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=20
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"❌ API Error for name={name}: {e}")
            return f"ERROR: {e}"

    def already_labeled(self, peerid: int) -> bool:
        self.cursor.execute("SELECT 1 FROM labels WHERE peerid = ?", (peerid,))
        return self.cursor.fetchone() is not None

    def run(self):
        df = pd.read_csv(self.input_path)
        total = len(df)
        self.logger.info(f"شروع برچسب‌گذاری {total} گروه")

        batch = []

        for idx, row in df.iterrows():
            peerid = row["peerid"]
            if self.already_labeled(peerid):
                continue

            label = self.get_topic_label(row["peer_name"], row["about"])
            batch.append((peerid, row["peer_name"], row["about"], label))

            if len(batch) >= self.batch_size:
                self._save_batch(batch)
                self.logger.info(f"✅ Batch ذخیره شد تا index {idx}")
                batch = []
                time.sleep(self.sleep_time)

        if batch:
            self._save_batch(batch)
            self.logger.info("✅ Batch نهایی ذخیره شد.")

        self.conn.close()
        self.logger.info("🎉 عملیات برچسب‌گذاری کامل شد.")

    def _save_batch(self, batch_data):
        self.cursor.executemany(
            "INSERT OR REPLACE INTO labels (peerid, peer_name, about, label) VALUES (?, ?, ?, ?)",
            batch_data
        )
        self.conn.commit()


# 🎯 اجرای مستقیم
if __name__ == "__main__":
    print("🔐 API Key:", os.getenv("OPENAI_API_KEY"))
    load_dotenv()
    labeler = BatchLabeler(
        api_key=os.getenv("OPENAI_API_KEY"),
        input_path="data/processed/groups_clean.csv",
        db_path="results/labels.db",
        batch_size=50,
        sleep_time=2
    )
    labeler.run()
