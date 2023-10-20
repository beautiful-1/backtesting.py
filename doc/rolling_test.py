import pandas as pd

# 创建一个示例时间序列数据
data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9])

# 创建一个滑动窗口对象，计算 3 个数据点的移动平均
rolling_window = data.rolling(window=3)

# 计算移动平均
moving_average = rolling_window.mean()
print(moving_average)
