import torch

print("=== PyTorch Tensor Demo ===")

a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print("Tensor a:", a)
print("Tensor b:", b)

print("a + b =", a + b)
print("a * 2 =", a * 2)
print("Dot product =", torch.dot(a, b))
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


print("\n=== Linear Regression Demo ===")

# 1. 加载 Diabetes 数据集
X, y = load_diabetes(return_X_y=True)

print("Dataset shape:", X.shape)
print("Target shape:", y.shape)

# 2. 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:", len(X_train))
print("Test samples:", len(X_test))

# 3. 创建线性回归模型
model = LinearRegression()

# 4. 使用训练集训练模型
model.fit(X_train, y_train)

# 5. 使用测试集进行预测
y_pred = model.predict(X_test)

# 6. 计算 MAE 和 MSE
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print("\n=== Prediction vs Actual ===")

for i in range(10):
    error = abs(y_test[i] - y_pred[i])

    print(
        f"Sample {i + 1}: "
        f"Actual = {y_test[i]:.2f}, "
        f"Predicted = {y_pred[i]:.2f}, "
        f"Absolute Error = {error:.2f}"
    )

print("\n=== Conclusion ===")
print(
    f"The model's predictions differ from the actual values "
    f"by about {mae:.2f} target-value units on average."
)