import pandas as pd
import numpy as np
import os

# === 設定 ===
input_csv = 'test_synthetic_linear_sensitivity.csv'
output_dir = 'split_groups_by_single_feature'
os.makedirs(output_dir, exist_ok=True)

# === 讀取資料 ===
df = pd.read_csv(input_csv)
feature_cols = [col for col in df.columns if col.startswith('X')]
num_features = len(feature_cols)
tolerance = 1e-6

# === Step 1: 先嘗試用固定特徵 group ===
# 排序資料（避免不同順序影響）
df_sorted = df.sort_values(by=feature_cols).reset_index(drop=True)

# === Step 2: 嘗試用 sliding window 分組 ===
groups = []
used = np.zeros(len(df_sorted), dtype=bool)

for i in range(len(df_sorted)):
    if used[i]:
        continue
    row = df_sorted.iloc[i][feature_cols].values
    # 找出與該 row 除了一個 Xi 外都相同的 row
    mask = []
    for j in range(len(df_sorted)):
        diff = np.abs(df_sorted.iloc[j][feature_cols].values - row)
        # 計算有幾個值超過 tolerance
        diff_count = np.sum(diff > tolerance)
        if diff_count <= 1:
            mask.append(j)
    mask = sorted(mask)
    used[mask] = True
    groups.append(df_sorted.iloc[mask])

# === Step 3: 分析每個 group 的變動特徵 ===
for idx, g in enumerate(groups):
    stds = g[feature_cols].std(axis=0).values
    varying_idx = np.argmax(stds)
    feature_name = feature_cols[varying_idx]
    filename = f"group_{idx+1:03d}_varying_{feature_name}.csv"
    g.to_csv(os.path.join(output_dir, filename), index=False)

print(f"✅ 分組完成，共輸出 {len(groups)} 組到 '{output_dir}'")