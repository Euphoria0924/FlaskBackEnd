import numpy as np
import pandas as pd

df = pd.read_csv('D:\PycharmProjects\FlaskBackEnd\\uploads\\upload_example.csv')
df = df.drop(columns=["TIME", "MW"])
df = df.dropna(subset=["Size"])
# 按照depth降序
df = df.sort_values(by=['BIT DEPTH'], ascending=True)
df = df[(df != 0).all(axis=1)]
df = df[(df != -999.25).all(axis=1)]
min_depth = df["BIT DEPTH"].min()
max_depth = df["BIT DEPTH"].max()
depth_range = np.arafnge(min_depth, max_depth + 0.2, 0.2)  # 等间隔深度
# 重建 DataFrame：以新的深度范围为基准，插入原始数据
df_resampled = pd.DataFrame({"DEPTH": depth_range})
# 将原始数据合并到新的深度范围上
df_resampled = pd.merge(df_resampled, df, on="DEPTH", how="left")
df_resampled = df_resampled.drop(columns=["DEPTH"])
# 对缺失值进行插值填充（线性插值）
df_resampled.interpolate(method="linear", inplace=True)
# 检查填充完成后的数据
print("\nResampled DataFrame with Interpolated Values:")
print(df_resampled)