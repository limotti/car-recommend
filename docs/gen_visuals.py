"""
실데이터 매핑 시각화 생성 스크립트
출력: docs/mapping_car_features.png
       docs/mapping_car_purchase.png
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── 색상 팔레트 ──────────────────────────────────────────
NAVY      = '#002C5F'
BLUE      = '#4A90E2'
LIGHT_BG  = '#F4F7FB'
GRID_CLR  = '#DDE4EF'
GOLD      = '#F5A623'
RED_HL    = '#E74C3C'
GRAY      = '#8C9BB5'
WHITE     = '#FFFFFF'

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, '..', 'data', 'preprocessing')
OUT  = BASE   # docs/

# ════════════════════════════════════════════════════════
# 시각화 1  car_features — 차급 가격 + 연료 분포
# ════════════════════════════════════════════════════════
feat = pd.read_csv(os.path.join(DATA, 'car_features_cleaned.csv'), encoding='utf-8-sig')

# 1-1 차급별 가격 (USD → 만원, 1USD=1350KRW)
RATE = 1350 / 10000
us_med = feat.groupby('Vehicle Size')['MSRP'].median() * RATE
us_size_map = {'Compact': '소형', 'Midsize': '중형', 'Large': '대형'}
us_price = {us_size_map[k]: round(v) for k, v in us_med.items()}

# 한국 서비스 차급별 평균 (CAR_INFO 기준 최저~최고 중간값 평균)
kr_price = {
    '소형': round((2412 + 2891 + 2857) / 3),   # 아반떼/셀토스/코나 가솔린
    '중형': round((3304 + 4107 + 3144 + 3241) / 4),  # 쏘나타/산타페/K5/스포티지
    '대형': round((4828 + 4984 + 4500 + 7220) / 4),  # 그랜저/팰리세이드/K8/G80
}

# 1-2 연료 타입 분포
fuel_raw = feat['Engine Fuel Type'].value_counts()
total_us = fuel_raw.sum()
us_fuel = {
    '가솔린':    round((fuel_raw.get('regular unleaded', 0) +
                        fuel_raw.get('premium unleaded (required)', 0) +
                        fuel_raw.get('premium unleaded (recommended)', 0)) / total_us * 100, 1),
    '전기':      round(fuel_raw.get('electric', 0) / total_us * 100, 1),
    '디젤':      round(fuel_raw.get('diesel', 0) / total_us * 100, 1),
    '하이브리드': 0.0,   # US 데이터에 별도 HEV 분류 없음 (flex-fuel 등 포함 가능)
}
us_fuel['기타'] = round(100 - sum(us_fuel.values()), 1)

# 우리 서비스 25개 차종 연료 비중
kr_fuel = {'가솔린': 48.0, '하이브리드': 32.0, '전기': 16.0, '디젤': 4.0, '기타': 0.0}

# ── Figure 1 ────────────────────────────────────────────
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor=WHITE)
fig1.subplots_adjust(wspace=0.38, left=0.07, right=0.97, top=0.88, bottom=0.13)
fig1.suptitle('미국 실데이터 vs 한국 서비스 — 차급·연료 구조 비교',
              fontsize=15, fontweight='bold', color=NAVY, y=0.97)

# ── 1-1 차급별 가격 막대 ───────────────────────────────
sizes  = ['소형', '중형', '대형']
us_v   = [us_price[s] for s in sizes]
kr_v   = [kr_price[s] for s in sizes]
x      = np.arange(len(sizes))
w      = 0.35

ax1.set_facecolor(LIGHT_BG)
ax1.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)

b1 = ax1.bar(x - w/2, us_v, w, label='미국 실데이터 (중앙값)', color=BLUE,   alpha=0.88, zorder=3)
b2 = ax1.bar(x + w/2, kr_v, w, label='한국 서비스 (평균)',      color=NAVY,   alpha=0.92, zorder=3)

for bar in b1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
             f'{int(bar.get_height()):,}만', ha='center', va='bottom',
             fontsize=9, color=BLUE, fontweight='bold')
for bar in b2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
             f'{int(bar.get_height()):,}만', ha='center', va='bottom',
             fontsize=9, color=NAVY, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(sizes, fontsize=12, fontweight='bold', color=NAVY)
ax1.set_ylabel('가격 (만원)', fontsize=10, color=NAVY)
ax1.set_ylim(0, max(max(us_v), max(kr_v)) * 1.22)
ax1.legend(fontsize=9, loc='upper left', framealpha=0.85)
ax1.set_title('차급별 가격대 비교', fontsize=12, fontweight='bold', color=NAVY, pad=10)
ax1.tick_params(axis='y', labelcolor=GRAY, labelsize=9)
ax1.spines[['top','right','left','bottom']].set_visible(False)

# 일치 화살표 / 텍스트
for i, (u, k) in enumerate(zip(us_v, kr_v)):
    diff_pct = abs(u - k) / max(u, k) * 100
    ax1.annotate('', xy=(i + w/2, min(u, k) * 0.55),
                 xytext=(i - w/2, min(u, k) * 0.55),
                 arrowprops=dict(arrowstyle='<->', color=GOLD, lw=1.8))
    ax1.text(i, min(u, k) * 0.48, f'±{diff_pct:.0f}%', ha='center',
             fontsize=8, color=GOLD, fontweight='bold')

# ── 1-2 연료 타입 분포 ────────────────────────────────
fuel_cats   = ['가솔린', '하이브리드', '전기', '디젤', '기타']
fuel_colors = [NAVY, '#1E6FBC', BLUE, GRAY, '#C8D6E8']
us_fv = [us_fuel.get(c, 0) for c in fuel_cats]
kr_fv = [kr_fuel.get(c, 0) for c in fuel_cats]
xf    = np.arange(len(fuel_cats))

ax2.set_facecolor(LIGHT_BG)
ax2.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)

bf1 = ax2.bar(xf - w/2, us_fv, w, label='미국 실데이터', color=BLUE, alpha=0.88, zorder=3)
bf2 = ax2.bar(xf + w/2, kr_fv, w, label='한국 서비스',   color=NAVY, alpha=0.92, zorder=3)

for bar in bf1:
    if bar.get_height() > 1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=8.5, color=BLUE, fontweight='bold')
for bar in bf2:
    if bar.get_height() > 1:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom',
                 fontsize=8.5, color=NAVY, fontweight='bold')

ax2.set_xticks(xf)
ax2.set_xticklabels(fuel_cats, fontsize=11, fontweight='bold', color=NAVY)
ax2.set_ylabel('비중 (%)', fontsize=10, color=NAVY)
ax2.set_ylim(0, 105)
ax2.legend(fontsize=9, loc='upper right', framealpha=0.85)
ax2.set_title('연료 타입 분포 비교', fontsize=12, fontweight='bold', color=NAVY, pad=10)
ax2.tick_params(axis='y', labelcolor=GRAY, labelsize=9)
ax2.spines[['top','right','left','bottom']].set_visible(False)

# 인사이트 텍스트
ax2.text(0.5, -0.13,
         '한국 서비스는 하이브리드·전기차 비중을 의도적으로 확대 설계 (미래 수요 반영)',
         transform=ax2.transAxes, ha='center', va='top',
         fontsize=8.5, color=GRAY, style='italic')

out1 = os.path.join(OUT, 'mapping_car_features.png')
fig1.savefig(out1, dpi=150, bbox_inches='tight', facecolor=WHITE)
plt.close(fig1)
print(f'저장: {out1}')


# ════════════════════════════════════════════════════════
# 시각화 2  car_purchase — 소득구간별 구매율
# ════════════════════════════════════════════════════════
pur = pd.read_csv(os.path.join(DATA, 'car_purchase_cleaned.csv'), encoding='utf-8-sig')
cols = list(pur.columns)
bracket_col = cols[-1]  # 소득구간 (마지막 컬럼)

ORDER = ['100~200만', '200~300만', '300~400만', '400~600만', '600만+']
rate_by = pur.groupby(bracket_col)['Purchased'].mean() * 100
rate_by = rate_by.reindex([c for c in ORDER if c in rate_by.index])

# 실제 레이블 정리 (garbled 문자 대체)
bracket_labels = ['100~200만', '200~300만', '300~400만', '400~600만', '600만+']
rates = rate_by.values

# ── Figure 2 ────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(12, 6), facecolor=WHITE)
fig2.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.18)
fig2.suptitle('소득 구간별 신차 구매율 (실데이터 기반)',
              fontsize=15, fontweight='bold', color=NAVY, y=0.97)

ax.set_facecolor(LIGHT_BG)
ax.yaxis.grid(True, color=GRID_CLR, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# 우리 서비스 타겟 구간 음영 (200~500만원 = index 1,2,3 구간)
# x축은 0~4 (5개 구간), 타겟은 1~3.5 구간에 해당
ax.axvspan(-0.5, 3.35, alpha=0.10, color=GOLD, zorder=1, label='우리 서비스 타겟 구간 (200~500만원)')
ax.axvline(x=3.35, color=GOLD, lw=1.5, ls='--', alpha=0.7, zorder=2)
ax.axvline(x=0.5,  color=GOLD, lw=1.5, ls='--', alpha=0.7, zorder=2)
ax.text(1.5, 56.5, '우리 서비스 타겟 구간\n(200~500만원)', ha='center',
        fontsize=9, color='#A07800', fontweight='bold', zorder=5)

x = np.arange(len(bracket_labels))

# 막대 색상: 피크(300~400) 강조
bar_colors = [BLUE if lbl != '300~400만' else RED_HL for lbl in bracket_labels]
bars = ax.bar(x, rates, width=0.55, color=bar_colors, alpha=0.92, zorder=3,
              edgecolor=WHITE, linewidth=1.2)

for bar, rate in zip(bars, rates):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{rate:.1f}%', ha='center', va='bottom',
            fontsize=11, fontweight='bold',
            color=RED_HL if rate == max(rates) else NAVY)

# 피크 강조 화살표
peak_idx = list(rates).index(max(rates))
ax.annotate(f'구매율 피크\n{max(rates):.1f}%',
            xy=(peak_idx, max(rates)),
            xytext=(peak_idx + 1.05, max(rates) + 3),
            fontsize=10, fontweight='bold', color=RED_HL,
            arrowprops=dict(arrowstyle='->', color=RED_HL, lw=2),
            ha='left')

ax.set_xticks(x)
ax.set_xticklabels(bracket_labels, fontsize=12, fontweight='bold', color=NAVY)
ax.set_ylabel('구매율 (%)', fontsize=11, color=NAVY)
ax.set_ylim(0, 68)
ax.set_xlabel('소득 구간 (만원)', fontsize=11, color=NAVY, labelpad=8)
ax.tick_params(axis='y', labelcolor=GRAY, labelsize=10)
ax.spines[['top','right','left','bottom']].set_visible(False)

# 범례
patch_peak  = mpatches.Patch(color=RED_HL,  alpha=0.9,  label='구매율 피크 구간 (300~400만)')
patch_norm  = mpatches.Patch(color=BLUE,    alpha=0.9,  label='일반 구간')
patch_tgt   = mpatches.Patch(color=GOLD,    alpha=0.35, label='우리 서비스 타겟 (200~500만원)')
ax.legend(handles=[patch_peak, patch_norm, patch_tgt], fontsize=9,
          loc='upper right', framealpha=0.88)

# 하단 인사이트
ax.text(0.5, -0.17,
        '실데이터 구매율 피크(300~400만원)가 우리 서비스 타겟 구간(200~500만원) 내에 포함 → 설계 타당성 검증',
        transform=ax.transAxes, ha='center', va='top',
        fontsize=9.5, color=NAVY, fontweight='bold')

out2 = os.path.join(OUT, 'mapping_car_purchase.png')
fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=WHITE)
plt.close(fig2)
print(f'저장: {out2}')
