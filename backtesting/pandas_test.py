import pandas as pd
import numpy as np

# 创建一个示例 Pandas Series
data = [10, 0, 0, 20, 30, 0, 0, 0, 40, 50]
index = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
dd = pd.Series(data, index=index)

# 寻找 Pandas Series 中等于 0 的元素的索引
zero_indices = (dd == 0).values.nonzero()[0]
print(type((dd == 0).values))
print((dd == 0).values)
ss =dd.values
print(dd.values.nonzero())
print(zero_indices)
# 获取 Pandas Series 的长度
length = len(dd)

# 使用 np.r_ 和 np.unique 组合生成 iloc
iloc = np.unique(np.r_[zero_indices, length - 1])

# 打印结果
print(iloc)
