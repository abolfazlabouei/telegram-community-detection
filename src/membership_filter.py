import pandas as pd
from typing import Set

class MembershipFilter:
    def __init__(self, membership_path: str, valid_group_ids: Set[int]):
        """
        :param membership_path: مسیر فایل عضویت
        :param valid_group_ids: لیست گروه‌های معتبر که از groups_clean گرفته شده‌اند
        """
        self.membership_path = membership_path
        self.valid_group_ids = valid_group_ids
        self.membership_df: pd.DataFrame = pd.DataFrame()
        self.filtered_df: pd.DataFrame = pd.DataFrame()

    def load_membership(self):
        self.membership_df = pd.read_csv(self.membership_path)

    def filter_by_valid_groups(self):
        if self.membership_df.empty:
            raise ValueError("Membership data is not loaded.")

        self.filtered_df = self.membership_df[
            self.membership_df["groupID"].isin(self.valid_group_ids)
        ]

    def report(self):
        total_before = self.membership_df["groupID"].nunique()
        total_after = self.filtered_df["groupID"].nunique()
        print(f"📊 تعداد گروه‌ها قبل از فیلتر: {total_before}")
        print(f"✅ تعداد گروه‌ها بعد از فیلتر: {total_after}")
        print(f"❌ چند گروه حذف شدند؟ {total_before - total_after}")

    def save_filtered(self, path: str):
        if self.filtered_df.empty:
            raise ValueError("Filtered data is empty.")
        self.filtered_df.to_csv(path, index=False)
