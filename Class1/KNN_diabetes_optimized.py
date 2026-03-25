import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_diabetes_knn():
    """
    优化后kNN算法对糖尿病数据集进行分类
    1. 缺失值处理：将生理上不合理的 0 值替换为特征的中位数。
    2. 超参数调优：使用 GridSearchCV 寻找最佳的 K 值和距离权重策略。
    3. 特征缩放：使用 StandardScaler 确保所有特征在同一量级。
    """
    print("--- 优化后糖尿病数据集kNN分类 ---")
    
    # 1. 加载数据
    data_path = os.path.join(os.path.dirname(__file__),'data/diabetes.csv')
    if not os.path.exists(data_path):
        print(f"错误: 找不到目录 {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # 2. 缺失值处理
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zeros:
        df[col] = df[col].replace(0, np.nan)
        # 使用该列的中位数填充缺失值
        df[col] = df[col].fillna(df[col].median())

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 3. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 超参数调优
    param_grid = {
        'n_neighbors': range(1, 31),             # 测试 K 从 1 到 30
        'weights': ['uniform', 'distance'],      # 测试等权重 vs 按距离加权
        'metric': ['euclidean', 'manhattan']     # 测试欧氏距离 vs 曼哈顿距离
    }

    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"最佳参数组合: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")

    # 使用最佳模型
    best_knn = grid_search.best_estimator_

    # 6. 预测并评估
    y_pred = best_knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n优化后准确率: {accuracy:.4f}")
    
    # 7. 保存结果
    results_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Correct': (y_test.values == y_pred)
    })
    
    output_path = os.path.join(os.path.dirname(__file__),'KNN_diabetes_results_optimized.csv')
    results_df.to_csv(output_path, index=False)
    print(f"优化后分类结果已保存至: {output_path}")

    # 8. 打印详细分类报告
    print("\n优化后分类结果报告:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    run_diabetes_knn()
