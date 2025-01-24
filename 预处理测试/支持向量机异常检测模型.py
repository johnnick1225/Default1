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

class 完整拟合模型:
    def __init__(self, 文件路径, 结果目录='预处理测试/results/charts', SVM异常值比例=0.07):
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
        self.SVM核函数 = 'sigmoid'  # rbf, linear, poly, sigmoid
        self.SVM异常值比例 = SVM异常值比例

    def 数据预处理(self):
        # 获取当前时间
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 读取数据
        df = pd.read_excel(self.文件路径)
        
        # 划分特征和标签
        X = df.iloc[:, :3].copy()  # 前三列为自变量
        y = df.iloc[:, 3:6].copy()  # 后三列为因变量
        
        # 异常值检测
        svm = OneClassSVM(
            kernel=self.SVM核函数,
            nu=self.SVM异常值比例
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
        
        # 绘图
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体
        plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
        
        # 设置图片保存目录
        photos_dir = 'H:/VS Code/Default path/预处理测试/results/photos'
        os.makedirs(photos_dir, exist_ok=True)
        
        # 1. 前处理前后散点图对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # 读取原始数据用于绘图
        df_original = pd.read_excel(self.文件路径)
        X_original = df_original.iloc[:, :3]
        
        # 前处理前
        ax1.scatter(X_original.iloc[:, 0], X_original.iloc[:, 1], c='blue', alpha=0.5)
        ax1.scatter(X_original.iloc[self.原始异常值索引, 0], X_original.iloc[self.原始异常值索引, 1],
                   c='red', marker='x', s=100, alpha=0.8)
        ax1.set_title('前处理前散点图')
        ax1.set_xlabel(X_original.columns[0])
        ax1.set_ylabel(X_original.columns[1])
        
        # 前处理后
        ax2.scatter(self.X_train.iloc[:, 0], self.X_train.iloc[:, 1], c='green', alpha=0.5)
        ax2.set_title('前处理后散点图')
        ax2.set_xlabel(self.X_train.columns[0])
        ax2.set_ylabel(self.X_train.columns[1])
        
        plt.tight_layout()
        plt.savefig(os.path.join(photos_dir, f'支持向量机sigmoid异常值比例{self.SVM异常值比例}前处理对比_{current_time}.png'))
        plt.close()
        
        
        # 2. 三个因变量的三维散点图
        fig = plt.figure(figsize=(18, 6))
        
        # 读取原始数据用于绘图
        df_original = pd.read_excel(self.文件路径)
        X_original = df_original.iloc[:, :3]
        y_original = df_original.iloc[:, 3:6]
        
        # 定义不同颜色映射
        cmaps = ['viridis', 'plasma', 'inferno']
        
        for i, col in enumerate(y_original.columns):
            ax = fig.add_subplot(1, 3, i+1, projection='3d')
            
            # 绘制所有点（包括异常值）
            sc = ax.scatter(
                X_original.iloc[:, 0],
                X_original.iloc[:, 1],
                X_original.iloc[:, 2],
                c=y_original[col],
                cmap=cmaps[i],
                alpha=0.5
            )
            
            # 用红叉标注异常值
            ax.scatter(
                X_original.iloc[self.原始异常值索引, 0],
                X_original.iloc[self.原始异常值索引, 1],
                X_original.iloc[self.原始异常值索引, 2],
                c='red', marker='x', s=100, alpha=0.8
            )
            
            ax.set_title(f'{col} 三维散点图（含异常值）')
            ax.set_xlabel(X_original.columns[0])
            ax.set_ylabel(X_original.columns[1])
            ax.set_zlabel(X_original.columns[2])
            fig.colorbar(sc, label=col)
            
        plt.tight_layout()
        plt.savefig(os.path.join(photos_dir, f'支持向量机sigmoid异常值比例{self.SVM异常值比例}三维散点图_{current_time}.png'))
        plt.close()

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
        result_path = os.path.join(self.结果目录, f'支持向量机sigmoid_异常值比例{self.SVM异常值比例}_{current_time}.csv')
        
        metrics = []
        metrics.append([f"异常值比例: {self.SVM异常值比例}"])
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
        print(f"训练和测试结果已保存到: {result_path}")

    def 运行(self):
        self.数据预处理()
        self.构建模型()
        self.训练模型()
        self.评估模型()

# 使用示例
if __name__ == "__main__":
    模型 = 完整拟合模型('拟合模型/第一批.xlsx')
    模型.运行()
