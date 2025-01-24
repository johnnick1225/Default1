# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
from datetime import datetime
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

class 贝叶斯优化完整拟合模型:
    def __init__(self, 文件路径, 结果目录='预处理测试/results/charts'):
        # 初始化参数
        self.文件路径 = 文件路径
        self.结果目录 = 结果目录
        os.makedirs(self.结果目录, exist_ok=True)
        
        # 定义贝叶斯优化搜索空间
        self.搜索空间 = {
            '优化器选择': Integer(1, 3),
            '最大训练轮数': Integer(100, 1000),
            '批量大小': Integer(16, 128),
            '测试集比例': Real(0.1, 0.3),
            '随机种子': Integer(1, 100),
            'SVM核函数': Categorical(['rbf', 'linear', 'poly', 'sigmoid']),
            'SVM异常值比例': Real(0.01, 0.1)
        }

    def 数据预处理(self, 参数):
        # 获取当前时间
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 读取数据
        df = pd.read_excel(self.文件路径)
        
        # 划分特征和标签
        X = df.iloc[:, :3].copy()  # 前三列为自变量
        y = df.iloc[:, 3:6].copy()  # 后三列为因变量
        
        # 异常值检测
        svm = OneClassSVM(
            kernel=参数['SVM核函数'],
            nu=参数['SVM异常值比例']
        )
        异常值索引 = svm.fit_predict(X) == -1
        # 保存原始异常值索引用于绘图
        self.原始异常值索引 = 异常值索引.copy()
        # 重新索引数据
        X = X.loc[~异常值索引].copy().reset_index(drop=True)
        y = y.loc[~异常值索引].copy().reset_index(drop=True)
        # 重新索引异常值索引
        异常值索引 = pd.Series(异常值索引[~异常值索引]).reset_index(drop=True)
        
        # 划分训练测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=参数['测试集比例'],
            random_state=参数['随机种子']
        )
        
        # 数据标准化
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X_train_scaled = self.scaler_X.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler_X.transform(self.X_test)
        self.y_train_scaled = self.scaler_y.fit_transform(self.y_train)
        self.y_test_scaled = self.scaler_y.transform(self.y_test)

    def 构建模型(self, 参数):
        self.model = Sequential([
            Dense(64, input_dim=self.X_train_scaled.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(3)  # 输出层，3个因变量
        ])
        
        # 设置优化器
        if 参数['优化器选择'] == 1:
            optimizer = 'adam'
        elif 参数['优化器选择'] == 2:
            from tensorflow.keras.optimizers import SGD
            optimizer = SGD(learning_rate=0.01)
        elif 参数['优化器选择'] == 3:
            from tensorflow.keras.optimizers import RMSprop
            optimizer = RMSprop(learning_rate=0.01)
        else:
            raise ValueError("无效的优化器选择，请选择 1 (Adam), 2 (SGD), 或 3 (RMSprop)")
        
        self.model.compile(optimizer=optimizer, loss='mse')

    def 训练模型(self, 参数):
        early_stopping = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
        self.history = self.model.fit(
            self.X_train_scaled, self.y_train_scaled,
            epochs=参数['最大训练轮数'],
            batch_size=参数['批量大小'],
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
        
        # 返回测试集平均RMSE作为优化目标
        return np.mean([v['RMSE'] for v in self.测试集指标.values()])

    def 目标函数(self, 参数):
        self.数据预处理(参数)
        self.构建模型(参数)
        self.训练模型(参数)
        return self.评估模型()

    def 运行(self):
        from skopt import gp_minimize
        from skopt.utils import use_named_args

        # 将搜索空间转换为带名称的列表格式
        搜索空间列表 = [
            Integer(1, 3, name='优化器选择'),
            Integer(100, 1000, name='最大训练轮数'),
            Integer(16, 128, name='批量大小'),
            Real(0.1, 0.3, name='测试集比例'),
            Integer(1, 100, name='随机种子'),
            Categorical(['rbf', 'linear', 'poly', 'sigmoid'], name='SVM核函数'),
            Real(0.01, 0.1, name='SVM异常值比例')
        ]

        # 定义目标函数
        @use_named_args(搜索空间列表)
        def 目标函数包装器(**参数):
            return self.目标函数(参数)

        # 运行贝叶斯优化
        优化结果 = gp_minimize(
            目标函数包装器,
            搜索空间列表,
            n_calls=50,
            random_state=42,
            verbose=True
        )

        # 保存最佳参数
        self.最佳参数 = {
            '优化器选择': 优化结果.x[0],
            '最大训练轮数': 优化结果.x[1],
            '批量大小': 优化结果.x[2],
            '测试集比例': 优化结果.x[3],
            '随机种子': 优化结果.x[4],
            'SVM核函数': 优化结果.x[5],
            'SVM异常值比例': 优化结果.x[6]
        }

        # 使用最佳参数重新训练模型
        self.数据预处理(self.最佳参数)
        self.构建模型(self.最佳参数)
        self.训练模型(self.最佳参数)
        self.评估模型()

        # 保存最佳参数和结果
        self.保存结果()

    def 保存结果(self):
        # 保存评价指标
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(self.结果目录, f'贝叶斯优化支持向量机_{current_time}.csv')
        
        metrics = []
        metrics.append(["最佳参数"])
        for k, v in self.最佳参数.items():
            metrics.append([f"{k}: {v}"])
        
        metrics.append([])
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
        print(f"贝叶斯优化结果已保存到: {result_path}")

# 使用示例
if __name__ == "__main__":
    模型 = 贝叶斯优化完整拟合模型('拟合模型/第一批.xlsx')
    模型.运行()