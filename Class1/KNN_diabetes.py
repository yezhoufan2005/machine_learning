import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def run_diabetes_knn():
    """使用 scikit-learn 实现糖尿病数据集分类"""
    print("--- 糖尿病数据集kNN分类 ---")
    
    # 1. 加载数据
    data_path = os.path.join(os.path.dirname(__file__),'data/diabetes.csv')
    if not os.path.exists(data_path):
        print(f"错误: 找不到目录 {data_path}")
        return

    df = pd.read_csv(data_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # 2. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 创建并训练模型
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_scaled, y_train)

    # 5. 预测并评估
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 6. 保存结果
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Correct': (y_test == y_pred)
    })
    
    output_path = os.path.join(os.path.dirname(__file__),'KNN_diabetes_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"分类结果已保存至: {output_path}")

    # 6. 打印分类报告
    print("\n分类结果报告:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    run_diabetes_knn()
