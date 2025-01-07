import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import os

# 参数设置
TEST_SIZE = 0.2  # 测试集比例
RANDOM_STATE = 42  # 随机种子
LOF_N_NEIGHBORS = 20  # LOF邻居数量
LOF_CONTAMINATION = 0.05  # 异常值比例

# 读取数据
df = pd.read_excel('拟合模型/第一批.xlsx')

# 异常值检测
def 检测异常值(数据):
    lof = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS,
        contamination=LOF_CONTAMINATION
    )
    return lof.fit_predict(数据.select_dtypes(include=['number'])) == -1

# 获取异常值索引
异常值索引 = 检测异常值(df)

# 删除异常值
df = df.loc[~异常值索引].copy()

# 划分特征和标签
X = df.iloc[:, :3]  # 前三列为自变量
y = df.iloc[:, 3:6]  # 后三列为因变量

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# 确保y_train和y_test是DataFrame
y_train = pd.DataFrame(y_train, columns=y.columns)
y_test = pd.DataFrame(y_test, columns=y.columns)

# 合并特征和标签：将特征矩阵X和标签y按列合并，方便保存为完整数据集
train_data = pd.concat([X_train, y_train], axis=1)  # 训练集
test_data = pd.concat([X_test, y_test], axis=1)  # 测试集

# 创建data目录
os.makedirs('拟合模型/data', exist_ok=True)

# 保存数据集
train_data.to_excel('拟合模型/data/第一批训练集.xlsx', index=False)
test_data.to_excel('拟合模型/data/第一批测试集.xlsx', index=False)
