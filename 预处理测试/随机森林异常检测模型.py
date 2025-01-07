# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime

class 完整拟合模型:
    def __init__(self, 文件路径, 结果目录='预处理测试/results'):
        # 初始化参数
        self.文件路径 = 文件路径
        self.结果目录 = 结果目录
        os.makedirs(self.结果目录, exist_ok=True)
        
        # 模型参数
        self.优化器选择 = 1  # 1->Adam, 2->SGD, 3->RMSprop
        self.最大训练轮数 = 1000
        self.批量大小 = 32
        self.测试集比例 = 0.2
        self.随机种子 = 42
        self.随机森林污染比例 = 0.05
        self.随机森林树数量 = 100

    def 数据预处理(self):
        # 读取数据
        df = pd.read_excel(self.文件路径)
        
        # 异常值检测
        iso_forest = IsolationForest(
            n_estimators=self.随机森林树数量,
            contamination=self.随机森林污染比例,
            random_state=self.随机种子
        )
        异常值索引 = iso_forest.fit_predict(df.select_dtypes(include=['number'])) == -1
        df = df.loc[~异常值索引].copy()
        
        # 划分特征和标签
        X = df.iloc[:, :3]  # 前三列为自变量
        y = df.iloc[:, 3:6]  # 后三列为因变量
        
        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.测试集比例,
            random_state=self.随机种子
        )
        
        # 数据标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)

    def 构建模型(self):
        self.model = Sequential([
            Dense(64, input_dim=self.X_train_scaled.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)  # 输出层，3个因变量
        ])
        
        # 设置优化器
        if self.优化器选择 == 1:
            optimizer = 'adam'
        elif self.优化器选择 == 2:
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(learning_rate=0.01)
        elif self.优化器选择 == 3:
            from tensorflow.keras.optimizers import RMSprop
            optimizer = RMSprop(learning_rate=0.01)
        else:
            raise ValueError("无效的优化器选择，请选择 1 (Adam), 2 (SGD), 或 3 (RMSprop)")
        
        self.model.compile(optimizer=optimizer, loss='mse')

    def 训练模型(self):
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train_scaled,
            epochs=self.最大训练轮数,
            batch_size=self.批量大小,
            verbose=1,
            callbacks=[early_stopping]
        )

    def 评估模型(self):
        # 预测并保存结果
        self.y_train_pred = self.scaler_y.inverse_transform(self.model.predict(self.X_train_scaled))
        self.y_test_pred = self.scaler_y.inverse_transform(self.model.predict(self.X_test_scaled))
        
        # 计算评价指标
        def 计算指标(y_true, y_pred):
            return {
                'MAE': mean_absolute_error(y_true, y_pred),
                'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
                'R2': r2_score(y_true, y_pred)
            }
        
        self.训练集指标 = {col: 计算指标(self.y_train[col], self.y_train_pred[:, i]) 
                        for i, col in enumerate(self.y_train.columns)}
        self.测试集指标 = {col: 计算指标(self.y_test[col], self.y_test_pred[:, i]) 
                        for i, col in enumerate(self.y_test.columns)}
        
        # 保存结果
        self.保存结果()

    def 保存结果(self):
        # 保存评价指标
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(self.结果目录, f'随机森林_{current_time}.csv')
        
        metrics = []
        metrics.append(["训练集指标"])
        for col in self.y_train.columns:
            metrics.append([
                f"变量: {col}",
                f"MAE: {self.训练集指标[col]['MAE']}",
                f"RMSE: {self.训练集指标[col]['RMSE']}",
                f"R2: {self.训练集指标[col]['R2']}"
            ])
        
        metrics.append([])
        metrics.append(["测试集指标"])
        for col in self.y_test.columns:
            metrics.append([
                f"变量: {col}",
                f"MAE: {self.测试集指标[col]['MAE']}",
                f"RMSE: {self.测试集指标[col]['RMSE']}",
                f"R2: {self.测试集指标[col]['R2']}"
            ])
        
        pd.DataFrame(metrics).to_csv(result_path, index=False, header=False)
        print(f"训练和测试结果已保存到: {result_path}")

    def 运行(self):
        self.数据预处理()
        self.构建模型()
        self.训练模型()
        self.评估模型()

# 使用示例
if __name__ == "__main__":
    模型 = 完整拟合模型('e:/VS Code/Default path/拟合模型/第一批.xlsx')
    模型.运行()
