# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM

file_path = 'E:/研究/数据整理/第一批.xlsx'
data = pd.read_excel(file_path)

data.columns = ['YT', 'DB', 'CB', 'XRC', 'HXG', 'JSY']

X = data[['YT', 'DB', 'CB']]
Y = data[['XRC', 'HXG', 'JSY']]

data = data.dropna()

xlim = (X['YT'].min(), X['YT'].max())
ylim = (X['DB'].min(), X['DB'].max())
zlim = (X['CB'].min(), 1.4)

fig = plt.figure(figsize=(20, 15))
colors = ['Reds', 'Blues', 'Greens']

for i, col in enumerate(Y.columns):
    ax_left = fig.add_subplot(3, 2, 2 * i + 1, projection='3d')
    sc_left = ax_left.scatter(X['YT'], X['DB'], X['CB'], c=Y[col], cmap=colors[i])
    ax_left.set_title(f'Before Outlier Removal ({col})')
    ax_left.set_xlabel('YT')
    ax_left.set_ylabel('DB')
    ax_left.set_zlabel('CB')
    ax_left.set_xlim(xlim)
    ax_left.set_ylim(ylim)
    ax_left.set_zlim(zlim)

    oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
    outlier_pred = oc_svm.fit_predict(X)
    outliers = outlier_pred == -1
    data_cleaned = data[~outliers]

    X_cleaned = data_cleaned[['YT', 'DB', 'CB']]
    Y_cleaned = data_cleaned[['XRC', 'HXG', 'JSY']]

    ax_right = fig.add_subplot(3, 2, 2 * i + 2, projection='3d')
    sc_right = ax_right.scatter(X_cleaned['YT'], X_cleaned['DB'], X_cleaned['CB'], c=Y_cleaned[col], cmap=colors[i])
    ax_right.set_title(f'After Outlier Removal ({col})')
    ax_right.set_xlabel('YT')
    ax_right.set_ylabel('DB')
    ax_right.set_zlabel('CB')
    ax_right.set_xlim(xlim)
    ax_right.set_ylim(ylim)
    ax_right.set_zlim(zlim)

    cbar_left = fig.colorbar(sc_left, ax=ax_left, shrink=0.6, aspect=10)
    cbar_left.set_label(f'{col} Before')
    cbar_right = fig.colorbar(sc_right, ax=ax_right, shrink=0.6, aspect=10)
    cbar_right.set_label(f'{col} After')

plt.tight_layout()
plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X_cleaned, Y_cleaned, test_size=0.2, random_state=42)

output_dir = r'E:\VS Code\Default path\第一批\data'
X_train.to_csv(f'{output_dir}\X_train.csv', index=False)
X_test.to_csv(f'{output_dir}\X_test.csv', index=False)
Y_train.to_csv(f'{output_dir}\Y_train.csv', index=False)
Y_test.to_csv(f'{output_dir}\Y_test.csv', index=False)

print("数据预处理完成，训练集和测试集已保存。")
