import pandas as pd

# 创建一个Series对象
data = pd.Series([10, 20, 30, 40, 50])
print("原始Series:")
print(data)

# 访问元素
print("第一个元素:", data[0])
print("前两个元素:")
print(data[:2])
print(data[2:])

# 自定义索引
data = pd.Series([10, 20, 30, 40, 50], index=['A', 'B', 'C', 'D', 'E'])
print("带有自定义索引的Series:")
print(data)

# 矢量化操作
print("加法操作:")
print(data + 5)

# 数据对齐
data1 = pd.Series([1, 2, 3], index=['A', 'B', 'C'])
data2 = pd.Series([10, 20, 30], index=['B', 'C', 'D'])
print("数据对齐:")
print(data1 + data2)

# 计算统计数据
print("均值:", data.mean())
print("最大值:", data.max())

# 缺失值处理
data3 = pd.Series([10, None, 30, 40, None])
print("带有缺失值的Series:")
print(data3)
print("缺失值填充:")
print(data3.fillna(0))

# 绘图
# import matplotlib.pyplot as plt
# data.plot(kind='bar')
# plt.show()
