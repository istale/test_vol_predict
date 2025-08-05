import pandas as pd
import numpy as np
import random

# === 基本參數 ===
num_features = 5
samples_per_group = 50
groups_per_feature = 3  # 每個 Xi 有幾組不同背景
slopes = [8, -2, 4.5, 1, -3]     # Xi 對 Y 的斜率
biases = [1, 2, 3, 4, 5]         # Xi 對 Y 的截距
fixed_value_range = (0.3, 0.7)   # 背景值固定區間

# === 合成資料 ===
all_rows = []

for xi in range(num_features):
    for group_id in range(groups_per_feature):
        # 隨機產生背景值（其他 Xj 固定）
        background = [round(random.uniform(*fixed_value_range), 3) for _ in range(num_features)]
        slope = slopes[xi]
        bias = biases[xi]

        # 產生 Xi 從 0~1 的變動
        xi_values = np.linspace(0, 1, samples_per_group)
        for val in xi_values:
            x_row = background.copy()
            x_row[xi] = val  # 僅 xi 改變
            y = slope * val + bias
            all_rows.append(x_row + [y])

# === 組裝成 DataFrame 並輸出 ===
columns = [f"X{i}" for i in range(num_features)] + ["Y"]
df = pd.DataFrame(all_rows, columns=columns)

df.to_csv("test_synthetic_linear_sensitivity.csv", index=False)
print("✅ 合成資料完成，儲存為 'test_synthetic_linear_sensitivity.csv'")