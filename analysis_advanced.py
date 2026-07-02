# -*- coding: utf-8 -*-
"""
서울 상권 매출 분석 — 심화편 (관계분석 + 예측)
- 심화1(관계): 거래 건수가 많으면 매출도 높은가? 업종별 '객단가'는?
- 심화2(예측): 거래 건수로 매출금액을 예측하는 회귀 모델 (머신러닝 입문)
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 한글 폰트
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
EOK = 100_000_000

# 1단계 · 불러오기 & 정리
df = pd.read_csv("서울시 상권분석서비스(추정매출-상권)_2025년.csv", encoding="cp949")
df = df[(df["당월_매출_금액"] > 0) & (df["당월_매출_건수"] > 0)].copy()

# ── 파생변수(피처 엔지니어링): 객단가 = 매출금액 / 매출건수 ──
df["객단가"] = df["당월_매출_금액"] / df["당월_매출_건수"]

# ═══════════════ 심화1 · 관계분석 ═══════════════
# (a) 거래건수 ↔ 매출금액 상관관계
corr = df["당월_매출_건수"].corr(df["당월_매출_금액"])
print(f"[관계] 거래건수 ↔ 매출금액 상관계수: {corr:.3f}  (1에 가까울수록 강한 정비례)")

# 산점도 (너무 큰 이상치는 잘라 가독성 확보: 상위 1% 제외)
q99_x = df["당월_매출_건수"].quantile(0.99)
q99_y = df["당월_매출_금액"].quantile(0.99)
sample = df[(df["당월_매출_건수"] <= q99_x) & (df["당월_매출_금액"] <= q99_y)].sample(3000, random_state=42)
plt.figure(figsize=(8,5))
plt.scatter(sample["당월_매출_건수"]/10000, sample["당월_매출_금액"]/EOK,
            alpha=.25, color=TEAL, s=18)
plt.title(f"거래건수 vs 매출금액 — 뚜렷한 정비례 (상관 {corr:.2f})", fontsize=13, fontweight="bold")
plt.xlabel("거래 건수 (만 건)"); plt.ylabel("매출금액 (억원)")
plt.tight_layout(); plt.savefig("charts/adv1_scatter.png", dpi=130); plt.show()

# (b) 업종별 평균 객단가 Top / Bottom
gu = df.groupby("서비스_업종_코드_명")["객단가"].mean().sort_values(ascending=False)
pick = pd.concat([gu.head(5), gu.tail(5)]) / 10000  # 만원 단위
plt.figure(figsize=(9,5))
colors = [AMBER]*5 + [TEAL]*5
sns.barplot(x=pick.values, y=pick.index, palette=colors)
plt.title("업종별 평균 객단가 — 상위 5 vs 하위 5", fontsize=13, fontweight="bold")
plt.xlabel("건당 결제액 (만원)"); plt.ylabel("")
plt.tight_layout(); plt.savefig("charts/adv2_unitprice.png", dpi=130); plt.show()

# ═══════════════ 심화2 · 예측(회귀) ═══════════════
# 거래건수(X)로 매출금액(y) 예측 — 가장 기본적인 선형회귀
X = df[["당월_매출_건수"]]
y = df["당월_매출_금액"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
coef = model.coef_[0]   # 건당 매출 증가폭 = 평균 객단가에 해당

print(f"[예측] 선형회귀 R² = {r2:.3f}  (1에 가까울수록 잘 맞춤)")
print(f"[예측] 거래 1건당 매출 약 {coef:,.0f}원 증가로 학습됨")

# 실제 vs 예측 산점도
plt.figure(figsize=(7,6))
plt.scatter(y_test/EOK, pred/EOK, alpha=.25, color=TEAL, s=16)
lim = np.quantile(y_test/EOK, 0.99)
plt.plot([0,lim],[0,lim], "--", color=AMBER, lw=2, label="완벽 예측선")
plt.xlim(0,lim); plt.ylim(0,lim)
plt.title(f"실제 매출 vs 예측 매출 (R²={r2:.2f})", fontsize=13, fontweight="bold")
plt.xlabel("실제 매출 (억원)"); plt.ylabel("예측 매출 (억원)"); plt.legend()
plt.tight_layout(); plt.savefig("charts/adv3_pred.png", dpi=130); plt.show()

# ═══════════════ 인사이트 ═══════════════
print("\n" + "="*50 + "\n심화 인사이트\n" + "="*50)
print(f"1) 거래건수와 매출금액은 상관 {corr:.2f}로 강한 정비례 → 매출의 핵심은 '방문·결제 횟수'")
print(f"2) 객단가 1위 업종: {gu.index[0]} (약 {gu.iloc[0]/10000:,.0f}만원) — 고가·저빈도 업종")
print(f"3) 거래건수만으로 매출을 R²={r2:.2f} 수준까지 설명 가능 → 간단 모델로도 예측력 확보")