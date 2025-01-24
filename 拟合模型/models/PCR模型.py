# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# === 变量设置 ===
data_dir = '拟合模型/data'
result_dir = '拟合模型/results'
os.makedirs(result_dir, exist_ok=True)

# === 数据加载 ===
train_data = pd.read_excel(os.path.join(data_dir, '第一批训练集.xlsx'))
test_data = pd.read_excel(os.path.join(data_dir, '第一批测试集.xlsx'))

# 划分特征和标签
X_train = train_data.iloc[:, :3]  # 前三列为自变量
Y_train = train_data.iloc[:, 3:6]  # 后三列为因变量
X_test = test_data.iloc[:, :3]
Y_test = test_data.iloc[:, 3:6]

# === 数据标准化 ===
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# === 构建PCR模型 ===
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

models = [LinearRegression() for _ in range(Y_train_scaled.shape[1])]

# === 训练模型 ===
for i, model in enumerate(models):
    print(f"训练第{i+1}个输出变量...")
    model.fit(X_train_pca, Y_train_scaled[:, i])

# === 预测 ===
Y_train_pred = np.array([model.predict(X_train_pca) for model in models]).T
Y_test_pred = np.array([model.predict(X_test_pca) for model in models]).T

# 反标准化
Y_train_pred = scaler_Y.inverse_transform(Y_train_pred)
Y_test_pred = scaler_Y.inverse_transform(Y_test_pred)

# === 计算评价指标 ===
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

train_metrics = {col: calculate_metrics(Y_train[col], Y_train_pred[:, i]) for i, col in enumerate(Y_train.columns)}
test_metrics = {col: calculate_metrics(Y_test[col], Y_test_pred[:, i]) for i, col in enumerate(Y_test.columns)}

# 创建图像保存目录
photo_dir = os.path.join(result_dir, 'photo')
os.makedirs(photo_dir, exist_ok=True)

# === 可视化预测结果 ===
for i, col in enumerate(Y_test.columns):
    plt.figure()
    plt.plot(Y_test[col], label='True')
    plt.plot(Y_test_pred[:, i], label='Predicted')
    plt.title(f'{col} 真实值与预测值对比')
    plt.legend()
    plt.savefig(os.path.join(photo_dir, f'{col}_comparison.png'))
    plt.close()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for i in range(min(9, len(X_test.columns) * len(Y_test.columns))):
    x_col = X_test.columns[i // len(Y_test.columns)]
    y_col = Y_test.columns[i % len(Y_test.columns)]
    axes[i].scatter(X_test[x_col], Y_test[y_col], label='True', color='blue', alpha=0.6)
    axes[i].scatter(X_test[x_col], Y_test_pred[:, i % len(Y_test.columns)], label='Predicted', color='red', alpha=0.6)
    axes[i].set_title(f'{x_col} 与 {y_col} 关系图')
    axes[i].set_xlabel('Independent Variable')
    axes[i].set_ylabel('Dependent Variable')
    axes[i].legend()

plt.tight_layout()
plt.savefig(os.path.join(photo_dir, 'feature_scatter.png'))
plt.close()

# === 保存评价指标 ===
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = os.path.join(result_dir, f'PCR模型_{current_time}.csv')
metrics = []
metrics.append(["Training Metrics"])
for col in Y_train.columns:
    metrics.append([
        f"Variable: {col}",
        f"MAE: {train_metrics[col][0]}",
        f"RMSE: {train_metrics[col][1]}",
        f"R2: {train_metrics[col][2]}"
    ])

metrics.append([])
metrics.append(["Testing Metrics"])
for col in Y_test.columns:
    metrics.append([
        f"Variable: {col}",
        f"MAE: {test_metrics[col][0]}",
        f"RMSE: {test_metrics[col][1]}",
        f"R2: {test_metrics[col][2]}"
    ])

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(result_path, index=False, header=False)

print(f"训练和测试结果已保存到指定路径: {result_path}")