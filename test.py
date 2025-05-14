import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel

# 读取数据
df = pd.read_csv('data_adjusted_pytorch.csv')

# 数据预处理
print("\n原始数据信息：")
print("=" * 50)
print(df.info())

print("\n缺失值统计：")
print("=" * 50)
print(df.isnull().sum())

# 处理缺失值和无穷大值
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(df.mean())

# 准备数据
X = df[['x1_optimized', 'x2_optimized', 'x3_optimized', 'x4_optimized', 'm_optimized']]
y = df['Y']

# 运行有序logit回归
model = OrderedModel(y, X, distr='logit')
results = model.fit()

# 打印模型结果
print("\n有序Logit回归结果：")
print("=" * 50)
print(results.summary())

# 提取并打印各个变量的系数
print("\n各变量系数：")
print("=" * 50)
for var, coef in zip(X.columns, results.params):
    print(f"{var}: {coef:.4f}")

# 计算并打印P值
print("\n各变量P值：")
print("=" * 50)
for var, p_value in zip(X.columns, results.pvalues):
    print(f"{var}: {p_value:.4f}")

# 计算并打印伪R方
print("\n模型拟合优度：")
print("=" * 50)
print(f"伪R方: {results.prsquared:.4f}")
