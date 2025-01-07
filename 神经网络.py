# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# === 变量设置 ===
data_dir = 'F:/8-Python/深度学习模型/第一批/data'
result_dir = 'F:/8-Python/深度学习模型/第一批/result'
os.makedirs(result_dir, exist_ok=True)

# 模型参数
optimizer_choice = 1  # 优化器选择：1->Adam, 2->SGD, 3->RMSprop
epochs = 1000          # 最大训练轮数
batch_size = 32       # 批量大小

# === 数据加载 ===
X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
Y_train = pd.read_csv(os.path.join(data_dir, 'Y_train.csv'))
X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
Y_test = pd.read_csv(os.path.join(data_dir, 'Y_test.csv'))

# === 数据标准化 ===
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

# === 构建神经网络模型 ===
model = Sequential([
    Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dense(32, activation='relu'),
    Dense(3)  # 输出层，3个因变量
])

# === 根据用户选择设置优化器 ===
if optimizer_choice == 1:
    optimizer = 'adam'
elif optimizer_choice == 2:
    from tensorflow.keras.optimizers import SGD
    optimizer = SGD(learning_rate=0.01)
elif optimizer_choice == 3:
    from tensorflow.keras.optimizers import RMSprop
    optimizer = RMSprop(learning_rate=0.01)
else:
    raise ValueError("无效的优化器选择，请选择 1 (Adam), 2 (SGD), 或 3 (RMSprop)")

model.compile(optimizer=optimizer, loss='mse')

# === 训练模型，添加早停机制 ===
early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
model.fit(X_train_scaled, Y_train_scaled, epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stopping])

# === 预测 ===
Y_train_pred = scaler_Y.inverse_transform(model.predict(X_train_scaled))
Y_test_pred = scaler_Y.inverse_transform(model.predict(X_test_scaled))

# === 计算评价指标 ===
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

train_metrics = {col: calculate_metrics(Y_train[col], Y_train_pred[:, i]) for i, col in enumerate(Y_train.columns)}
test_metrics = {col: calculate_metrics(Y_test[col], Y_test_pred[:, i]) for i, col in enumerate(Y_test.columns)}

# === 可视化预测结果 ===
for i, col in enumerate(Y_test.columns):
    plt.figure()
    plt.plot(Y_test[col], label='True')
    plt.plot(Y_test_pred[:, i], label='Predicted')
    plt.title(f'Comparison for {col}')
    plt.legend()
    plt.show()

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()
for i in range(min(9, len(X_test.columns) * len(Y_test.columns))):
    x_col = X_test.columns[i // len(Y_test.columns)]
    y_col = Y_test.columns[i % len(Y_test.columns)]
    axes[i].scatter(X_test[x_col], Y_test[y_col], label='True', color='blue', alpha=0.6)
    axes[i].scatter(X_test[x_col], Y_test_pred[:, i % len(Y_test.columns)], label='Predicted', color='red', alpha=0.6)
    axes[i].set_title(f'{x_col} - {y_col}')
    axes[i].set_xlabel('Independent Variable')
    axes[i].set_ylabel('Dependent Variable')
    axes[i].legend()

plt.tight_layout()
plt.show()

# === 保存评价指标 ===
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
result_path = os.path.join(result_dir, f'神经网络_{current_time}.csv')
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
