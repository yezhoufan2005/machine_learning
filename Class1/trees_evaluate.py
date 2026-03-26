import pandas as pd
import numpy as np
import os
import random
import matplotlib.pyplot as plt
from trees_decision import createTree, classify
from tree_plotter import createPlot

# 设置中文字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_titanic_data():
    """加载 Titanic 数据集"""
    data_path = os.path.join(os.path.dirname(__file__),'data/titanic8120/train.csv')
    df = pd.read_csv(data_path)
    
    # 简单特征选择
    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    df = df[features + ['Survived']]
    
    # 填充缺失值
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # 转换离散特征
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    
    # 将类别标签移到最后一列
    cols = df.columns.tolist()
    new_df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']]
    labels = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    return new_df.values.tolist(), labels

def load_wine_data():
    """加载 Wine数据集"""
    data_path=os.path.join(os.path.dirname(__file__),'data/wine/wine.data')
    df=pd.read_csv(data_path,header=None)

    # Wine数据集标准字段名
    wine_labels=[
        "Alcohol","Malic acid","Ash","Alcalinity of ash","Magnesium",
        "Total phenols","Flavanoids","Nonflavanoid phenols","Proanthocyanins",
        "Color intensity","Hue","OD280/OD315","Proline"
    ]

    # 将类别标签移到最后一列
    cols=df.columns.tolist()
    new_df=df[cols[1:]+[cols[0]]]

    return new_df.values.tolist(),wine_labels

def evaluate_accuracy(tree, labels, test_data):
    """计算准确率"""
    correct = 0
    for vec in test_data:
        prediction = classify(tree, labels, vec[:-1])
        if prediction == vec[-1]:
            correct += 1
    return correct / len(test_data) if len(test_data) > 0 else 0

def hold_out_validation(data, labels, method='C4.5', ratio=0.7):
    """留出法验证"""
    random.shuffle(data)
    split_idx = int(len(data) * ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    tree = createTree(train_data, labels[:], method=method)
    acc = evaluate_accuracy(tree, labels, test_data)
    return acc

def cross_validation(data, labels, method='C4.5', k=5):
    """交叉验证法"""
    random.shuffle(data)
    fold_size = len(data) // k
    accuracies = []
    
    for i in range(k):
        test_data = data[i*fold_size : (i+1)*fold_size]
        train_data = data[:i*fold_size] + data[(i+1)*fold_size:]
        
        tree = createTree(train_data, labels[:], method=method)
        acc = evaluate_accuracy(tree, labels, test_data)
        accuracies.append(acc)
        
    return np.mean(accuracies)

def plot_decision_tree(data, labels, method='C4.5', dataset_name=''):
    """绘制决策树图"""
    print(f"生成{dataset_name}数据集的{method}决策树图")
    tree = createTree(data, labels[:], method=method)
    createPlot(tree, title=f"{dataset_name} Decision Tree ({method})")

def run_experiments():
    datasets = [
        ("Titanic", load_titanic_data),
        ("Wine", load_wine_data)
    ]
    methods = ["C4.5", "CART"]
    
    print("="*50)
    print(f"{'Dataset':<10} | {'Method':<6} | {'Hold-out':<10} | {'Cross-Val':<10}")
    print("-"*50)
    
    for name, load_fn in datasets:
        data, labels = load_fn()
        for method in methods:
            # 留出法
            ho_acc = hold_out_validation(data, labels, method=method)
            # 5 折交叉验证
            cv_acc = cross_validation(data, labels, method=method, k=5)
            
            print(f"{name:<10} | {method:<6} | {ho_acc:<10.4f} | {cv_acc:<10.4f}")
    print("="*50)
    
    # 绘制决策树图
    print("\n绘制决策树图...")
    for name, load_fn in datasets:
        data, labels = load_fn()
        for method in methods:
            plot_decision_tree(data, labels, method=method, dataset_name=name)

if __name__ == '__main__':
    run_experiments()
