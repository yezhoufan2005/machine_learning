import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

def run_boston_regression():
    """波士顿房价数据集回归分析"""
    print("--- 波士顿房价回归分析 ---")
    
    # 1. 获取数据
    # 由于sklearn 1.2+已移除load_boston，使用fetch_openml代替
    print("从OpenML获取数据")
    boston = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    
    X = boston.data
    y = boston.target

    # 2. 数据划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 3. 特征缩放
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. 模型训练与评估
    models = {
        "多元线性回归": LinearRegression(),
        "Rideg回归": Ridge(alpha=1.0),
        "Lasso回归": Lasso(alpha=0.1)
    }

    results = []

    for name, model in models.items():
        # 训练
        model.fit(X_train_scaled, y_train)
        
        # 预测
        y_pred = model.predict(X_test_scaled)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results.append({
            "Model": name,
            "MSE": mse,
            "R2": r2
        })
        
        print(f"\n[{name}]")
        print(f"均方误差 (MSE): {mse:.4f}")
        print(f"决定系数 (R2 Score): {r2:.4f}")

    # 5. 结果汇总对比
    print("\n--- 回归结果汇总对比 ---")
    results_df = pd.DataFrame(results)
    print(results_df)

    # 6. 保存多元线性回归预测结果
    final_model = LinearRegression()
    final_model.fit(X_train_scaled, y_train)
    y_final_pred = final_model.predict(X_test_scaled)
    
    output_df = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_final_pred,
        'Error': y_test.values - y_final_pred
    })
    
    output_path = "boston_housing_results.csv"
    output_df.to_csv(output_path, index=False)
    print(f"\n预测结果已保存至: {output_path}")

if __name__ == '__main__':
    run_boston_regression()
