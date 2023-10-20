from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 加载示例数据集（鸢尾花数据集）
iris = load_iris()
print(iris)
X, y = iris.data, iris.target

# 将数据集分为训练集和测试集
"""
`random_state` 是许多机器学习算法和函数中的一个可选参数，用于控制随机性。当你在训练模型或执行某些操作时，通常会涉及到随机的元素，例如数据的随机分割、初始化权重的随机性等。`random_state` 参数允许你指定一个种子值，以确保在多次运行相同代码时，得到相同的随机结果，从而使结果可重现。

具体来说，当你在一个算法中使用 `random_state` 参数时，它会确定伪随机数生成器的种子值。这个种子值会影响所有随机性相关的操作。如果你在不同时间运行相同的代码，并提供相同的 `random_state` 值，你将获得相同的结果。

举个例子，当你使用 `train_test_split` 函数分割数据集为训练集和测试集时，你可以提供一个固定的 `random_state` 值，这样每次运行代码时，分割的结果都是相同的，这有助于在不同的试验中进行比较。

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

在上述代码中，`random_state=42` 将确保每次运行相同的代码都得到相同的训练集和测试集。

总之，`random_state` 是用来控制伪随机性的参数，有助于实验的可重复性和结果的可比较性。
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建 KNeighborsClassifier 模型
"""
`n_neighbors` 是 `KNeighborsClassifier` 中的一个超参数，代表K最近邻算法（K-Nearest Neighbors）中要考虑的最近邻居的数量。K最近邻算法是一种监督学习算法，通常用于分类问题，其核心思想是根据最近邻的样本的标签来对新样本进行分类。

`n_neighbors` 的大小会对模型的性能产生显著影响：

1. 当 `n_neighbors` 较小：如果你选择较小的值，比如1，模型将会对每个新样本选择最近的一个邻居来进行分类。这可能会导致模型对噪声敏感，因为它会受到单个数据点的影响。模型可能会过拟合训练数据，性能在训练集上很好，但在测试数据上表现较差。

2. 当 `n_neighbors` 较大：如果你选择较大的值，模型会考虑更多的邻居，这有助于平滑决策边界，减少模型的过拟合。然而，如果 `n_neighbors` 过大，模型可能会受到类别不平衡问题的影响，因为它会考虑到距离较远的邻居，这可能会导致模型在较小的类别上表现较差。

因此，选择适当的 `n_neighbors` 值取决于你的具体问题和数据集。你可以使用交叉验证等技术来帮助确定最佳的 `n_neighbors` 值，以获得最佳的模型性能。通常，通过尝试不同的 `n_neighbors` 值，从较小的值开始，然后逐渐增加，观察模型的性能，可以找到最适合你的数据的值。
"""
knn = KNeighborsClassifier(n_neighbors=3)  # 指定最近邻居的数量

# 在训练集上训练模型
knn.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn.predict(X_test)

# 计算分类准确率
accuracy = accuracy_score(y_test, y_pred)
print("分类准确率：", accuracy)


def example_function(arg1, *args, kwarg1=None, **kwargs):
    print("arg1:", arg1)
    print("Other args:", args)
    print("kwarg1:", kwarg1)
    print("Other kwargs:", kwargs)

# 调用示例
example_function(1, 2, 3, kwarg1="Hello", key1="Value1", key2="Value2")
