import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def plot_3d_scatter(file_path, output_prefix):
    # 读取Excel文件
    df = pd.read_excel(file_path)
    
    # 提取数据
    x = df.iloc[:, 0]
    y = df.iloc[:, 1] 
    z = df.iloc[:, 2]
    c1 = df.iloc[:, 3]
    c2 = df.iloc[:, 4]
    c3 = df.iloc[:, 5]
    
    # 创建3D图形
    fig = plt.figure(figsize=(18, 6))
    
    # 第一个子图
    ax1 = fig.add_subplot(131, projection='3d')
    scatter1 = ax1.scatter(x, y, z, c=c1, cmap='viridis')
    ax1.set_title('Variable 1')
    fig.colorbar(scatter1, ax=ax1, shrink=0.5)
    
    # 第二个子图
    ax2 = fig.add_subplot(132, projection='3d')
    scatter2 = ax2.scatter(x, y, z, c=c2, cmap='plasma')
    ax2.set_title('Variable 2')
    fig.colorbar(scatter2, ax=ax2, shrink=0.5)
    
    # 第三个子图
    ax3 = fig.add_subplot(133, projection='3d')
    scatter3 = ax3.scatter(x, y, z, c=c3, cmap='inferno')
    ax3.set_title('Variable 3')
    fig.colorbar(scatter3, ax=ax3, shrink=0.5)
    
    # 保存图像
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_3d_scatter.png')
    plt.close()

# 绘制第一批数据
plot_3d_scatter('第一批.xlsx', '第一批')

# 绘制第二批数据
plot_3d_scatter('第二批.xlsx', '第二批')
