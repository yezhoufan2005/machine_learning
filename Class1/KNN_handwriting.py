import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

def img2vector(filename):
    """将 32x32 的二进制图像文本文件转换为 1x1024 的向量"""
    returnVect = np.zeros((1, 1024))
    with open(filename, 'r') as fr:
        for i in range(32):
            lineStr = fr.readline()
            for j in range(32):
                returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect

def run_handwriting_knn():
    """使用 scikit-learn 实现手写数字图像数据集分类"""
    print("--- 手写数字图像数据集kNN分类 ---")
    
    # 1. 加载训练数据
    training_dir = os.path.join(os.path.dirname(__file__),'data/trainingDigits')
    if not os.path.exists(training_dir):
        print(f"错误: 找不到目录 {training_dir}")
        return

    training_file_list = os.listdir(training_dir)
    m = len(training_file_list)
    X_train = np.zeros((m, 1024))
    y_train = []

    for i in range(m):
        file_name_str = training_file_list[i]
        class_num = int(file_name_str.split('_')[0])
        y_train.append(class_num)
        X_train[i, :] = img2vector(os.path.join(training_dir, file_name_str))

    # 2. 加载测试数据
    test_dir = os.path.join(os.path.dirname(__file__),'data/testDigits')
    if not os.path.exists(test_dir):
        print(f"错误: 找不到目录 {test_dir}")
        return

    test_file_list = os.listdir(test_dir)
    m_test = len(test_file_list)
    X_test = np.zeros((m_test, 1024))
    y_test = []

    for i in range(m_test):
        file_name_str = test_file_list[i]
        class_num = int(file_name_str.split('_')[0])
        y_test.append(class_num)
        X_test[i, :] = img2vector(os.path.join(test_dir, file_name_str))

    # 3. 创建并训练模型
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # 4. 预测并评估
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 5. 保存结果
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Correct': (np.array(y_test) == y_pred)
    })
    
    output_path = os.path.join(os.path.dirname(__file__),'KNN_handwriting_results.csv')
    results_df.to_csv(output_path, index=False)
    print(f"分类结果已保存至: {output_path}")

    # 6. 打印分类报告
    print("\n分类结果报告:")
    print(classification_report(y_test, y_pred))

if __name__ == '__main__':
    run_handwriting_knn()
