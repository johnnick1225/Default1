import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

# 读取训练测试数据
train_data = pd.read_excel('拟合模型/data/第一批训练集.xlsx')
test_data = pd.read_excel('拟合模型/data/第一批测试集.xlsx')

# 划分特征和标签
X_train = train_data.iloc[:, :3].values
y_train = train_data.iloc[:, 3:6].values
X_test = test_data.iloc[:, :3].values
y_test = test_data.iloc[:, 3:6].values

# 创建深度神经网络模型
def create_model():
    model = Sequential()
    
    # 数据标准化层
    from tensorflow.keras.layers import Normalization
    norm_layer = Normalization()
    norm_layer.adapt(X_train)
    model.add(norm_layer)
    
    # 输入层
    model.add(Dense(256, input_dim=3, activation='relu', kernel_regularizer='l2'))
    
    # 隐藏层
    model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
    model.add(Dense(64, activation='relu', kernel_regularizer='l2'))
    
    # 输出层（3个输出对应3个因变量）
    model.add(Dense(3))
    
    # 编译模型
    model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    
    return model

# 训练模型
from tensorflow.keras.callbacks import EarlyStopping

model = create_model()

# 早停法
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    epochs=500, 
                    batch_size=64, 
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)

# 预测
y_pred = model.predict(X_test)

# 计算评价指标
def calculate_metrics(y_true, y_pred):
    metrics = {
        'r2': r2_score(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
    }
    return metrics

# 对每个因变量分别计算指标
results = {}
for i, col in enumerate(['行人侧', '呼吸高', '驾驶员']):
    results[col] = calculate_metrics(y_test[:, i], y_pred[:, i])

# 保存结果
os.makedirs('拟合模型/results', exist_ok=True)
with open('拟合模型/results/评价指标.txt', 'w', encoding='utf-8') as f:
    for col, metrics in results.items():
        f.write(f"{col} 评价指标:\n")
        f.write(f"  R2: {metrics['r2']:.4f}\n")
        f.write(f"  MAE: {metrics['mae']:.6f}\n")
        f.write(f"  RMSE: {metrics['rmse']:.6f}\n")
        f.write("\n")

print("模型训练完成，评价指标已保存到 results/评价指标.txt")
