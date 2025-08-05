import pandas as pd
import numpy as np
import os
import re
from sklearn.metrics import mean_squared_error

# === 設定 ===
input_dir = 'split_groups_by_single_feature'
output_summary = 'slope_bias_summary.csv'
tolerance = 1e-6

summary = []

# === 處理每個檔案 ===
for file in sorted(os.listdir(input_dir)):
    if file.endswith('.csv'):
        path = os.path.join(input_dir, file)
        df = pd.read_csv(path)

        # 從檔名中取出變動特徵名
        match = re.search(r'varying_(X\d+)', file)
        if not match:
            continue
        feature_name = match.group(1)
        xi_values = df[feature_name].values
        y_values = df['Y'].values

        # 線性迴歸（Y = slope * Xi + bias）
        A = np.vstack([xi_values, np.ones_like(xi_values)]).T
        slope, bias = np.linalg.lstsq(A, y_values, rcond=None)[0]
        y_pred = slope * xi_values + bias
        mse = mean_squared_error(y_values, y_pred)

        summary.append({
            'Group_File': file,
            'Varying_Feature': feature_name,
            'Slope': slope,
            'Bias': bias,
            'MSE': mse
        })

# === 匯出分析結果 ===
df_summary = pd.DataFrame(summary)
df_summary.to_csv(output_summary, index=False)
print(f"✅ 擬合分析完成，已儲存至：{output_summary}")