import pandas as pd
import re
from typing import Optional
import swifter

class GroupPreprocessor:
    def __init__(self, filepath: str, min_members: int = 200):
        self.filepath = filepath
        self.min_members = min_members
        self.df: Optional[pd.DataFrame] = None

    def load_and_filter(self):
        df = pd.read_csv(self.filepath, encoding='utf-8-sig', low_memory=False)
        df = df[df["peer_name"].notnull()]
        df = df[df["participants"] > self.min_members]
        df = df.drop_duplicates(subset="peerid")
        self.df = df

    def clean_text(self, text: str) -> str:
        if pd.isnull(text):
            return ""

        text = re.sub(r'http\S+|www\S+|t\.me/\S+|telegram\.me/\S+', '', text)
        text = re.sub(r'(\+98|0098|0)?9\d{9}', '', text)
        text = re.sub(r'@\w+', '', text)

        emoji_pattern = re.compile("["       
            u"\U0001F600-\U0001F64F"  
            u"\U0001F300-\U0001F5FF"  
            u"\U0001F680-\U0001F6FF"  
            u"\U0001F1E0-\U0001F1FF"  
            u"\U00002700-\U000027BF"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji_pattern.sub(r'', text)

        text = re.sub(r'[^\w\s\u0600-\u06FFa-zA-Z]', ' ', text)
        meaningful_short_words = {'آب', 'فن', 'مه', 'لب'}
        text = ' '.join([w for w in text.split() if len(w) >= 3 or w in meaningful_short_words])

        text = text.replace('ي', 'ی').replace('ك', 'ک')
        text = ' '.join(re.findall(r'[a-zA-Z\u0600-\u06FF]+', text))
        return text.strip()

    def is_really_persian(self, text: str) -> bool:
        if not isinstance(text, str) or text.strip() == "":
            return False
        persian_letters = "پچژگکیی"
        arabic_only_letters = "ةىيﻻـ"
        has_persian = any(c in text for c in persian_letters)
        has_arabic_only = any(c in text for c in arabic_only_letters)
        return has_persian and not has_arabic_only

    def clean_data(self):
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_filter() first.")

        self.df['peer_name'] = self.df['peer_name'].swifter.apply(self.clean_text)
        self.df['about'] = self.df['about'].swifter.apply(self.clean_text if 'about' in self.df else lambda x: "")

        self.df = self.df[
            self.df['peer_name'].swifter.apply(self.is_really_persian) |
            self.df['about'].swifter.apply(self.is_really_persian)
        ]

        self.df = self.df.dropna(subset=['peer_name', 'about'])
        self.df = self.df[
            (self.df['peer_name'].str.strip() != "") &
            (self.df['about'].str.strip() != "")
        ]

    def save(self, output_path: str):
        if self.df is not None:
            self.df.to_csv(output_path, index=False)
        else:
            raise ValueError("No data to save.")

if __name__=="__main__":
    pre = GroupPreprocessor(filepath="/mnt/d/Uni/thesis/data/idekav_subscription.csv")
    pre.load_and_filter()
    pre.clean_data()
    pre.save("data/processed/groups_clean.csv")