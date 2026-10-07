from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# 1. Load the Diabetes dataset
# 加载 Diabetes（糖尿病）数据集
diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

print("Dataset loaded successfully.")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# 2. Split the dataset into training and test sets
# 将数据划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nDataset split:")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")


# 3. Create and train the Linear Regression model
# 创建并训练 Linear Regression（线性回归）模型
model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression model trained successfully.")


# 4. Make predictions on the test set
# 使用测试集进行预测
y_pred = model.predict(X_test)


# 5. Calculate evaluation metrics
# 计算模型误差指标
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("\nModel Evaluation:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")


# 6. Compare predicted values with actual values
# 对比预测值和真实值
print("\nPrediction vs Actual:")
print("-" * 45)

for i in range(10):
    error = abs(y_test[i] - y_pred[i])

    print(
        f"Sample {i + 1:2d}: "
        f"Predicted = {y_pred[i]:7.2f}, "
        f"Actual = {y_test[i]:6.2f}, "
        f"Error = {error:6.2f}"
    )


# 7. Error conclusion
# 写出误差结论
print("\nConclusion:")
print(
    f"The model achieved an MAE of {mae:.2f}. "
    "This means that the predictions differ from the actual target values "
    "by about 43 units on average. "
    "The Linear Regression model provides a useful baseline, "
    "but some individual predictions still have relatively large errors."
)