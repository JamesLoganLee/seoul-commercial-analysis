# -*- coding: utf-8 -*-
"""
서울시 상권 추정매출 분석 (2025년)
- Q1. 어떤 업종의 매출이 가장 높은가?
- Q2. 매출이 높은 상권 Top 10은?
- Q3. 하루 중 언제 매출이 몰리는가? (시간대별)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ── 한글 폰트 (코랩=나눔 / 로컬=노토 자동 선택) ──
for p in ["/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
    if os.path.exists(p):
        fm.fontManager.addfont(p)
        plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", font=plt.rcParams["font.family"])
os.makedirs("charts", exist_ok=True)
TEAL, AMBER = "#1B8A6B", "#E2A12B"
EOK = 100_000_000   # 매출을 '억원' 단위로 바꾸기 위한 값

# 1단계 · 불러오기 (공공데이터라 encoding='cp949')
df = pd.read_csv("서울시_상권분석서비스_추정매출-상권__2025년.csv", encoding="cp949")
print("행 x 열:", df.shape)

# 2단계 · 살펴보기
print("\n[미리보기]\n", df[["상권_코드_명","서비스_업종_코드_명","당월_매출_금액"]].head())

# 3단계 · 전처리: 매출 결측/0 정리 (분석 대상은 매출>0)
df = df[df["당월_매출_금액"] > 0].copy()

# ── Q1. 업종별 총매출 Top 10 ──
by_type = (df.groupby("서비스_업종_코드_명")["당월_매출_금액"].sum()
             .sort_values(ascending=False).head(10) / EOK)
plt.figure(figsize=(9,4.8))
sns.barplot(x=by_type.values, y=by_type.index, color=TEAL)
plt.title("업종별 총매출 Top 10 (2025년)", fontsize=13, fontweight="bold")
plt.xlabel("총매출 (억원)"); plt.ylabel("")
plt.tight_layout(); plt.savefig("charts/q1_by_type.png", dpi=130); plt.close()

# ── Q2. 상권별 총매출 Top 10 ──
by_area = (df.groupby("상권_코드_명")["당월_매출_금액"].sum()
             .sort_values(ascending=False).head(10) / EOK)
plt.figure(figsize=(9,4.8))
sns.barplot(x=by_area.values, y=by_area.index, color=AMBER)
plt.title("매출 상위 상권 Top 10 (2025년)", fontsize=13, fontweight="bold")
plt.xlabel("총매출 (억원)"); plt.ylabel("")
plt.tight_layout(); plt.savefig("charts/q2_by_area.png", dpi=130); plt.close()

# ── Q3. 시간대별 총매출 ──
time_cols = {
    "00~06시":"시간대_00~06_매출_금액", "06~11시":"시간대_06~11_매출_금액",
    "11~14시":"시간대_11~14_매출_금액", "14~17시":"시간대_14~17_매출_금액",
    "17~21시":"시간대_17~21_매출_금액", "21~24시":"시간대_21~24_매출_금액",
}
by_time = pd.Series({label: df[col].sum()/EOK for label, col in time_cols.items()})
plt.figure(figsize=(9,4.5))
sns.barplot(x=by_time.index, y=by_time.values, color=TEAL)
plt.title("시간대별 총매출 — 언제 소비가 몰리나 (2025년)", fontsize=13, fontweight="bold")
plt.xlabel(""); plt.ylabel("총매출 (억원)")
plt.tight_layout(); plt.savefig("charts/q3_by_time.png", dpi=130); plt.close()

# 5단계 · 인사이트
print("\n" + "="*50 + "\n핵심 인사이트\n" + "="*50)
print(f"1) 매출 1위 업종: {by_type.index[0]} ({by_type.iloc[0]:,.0f}억원)")
print(f"2) 매출 1위 상권: {by_area.index[0]} ({by_area.iloc[0]:,.0f}억원)")
print(f"3) 매출 최대 시간대: {by_time.idxmax()} ({by_time.max():,.0f}억원)")
