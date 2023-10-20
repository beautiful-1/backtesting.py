import pandas as pd

# 创建一个示例 DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'Age': [25, 30, 35, 28, 22],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston'],
        'Salary': [60000, 80000, 75000, 70000, 65000]}

df = pd.DataFrame(data)

# head(): 显示 DataFrame 的前几行，默认是前5行
print("显示前三行")
print(df.head(3))  # 显示前3行

# tail(): 显示 DataFrame 的后几行，默认是后5行
print("显示后两行:")
print(df.tail(2))  # 显示后2行

# describe(): 提供统计信息的摘要，包括计数、均值、标准差、最小值、百分位数和最大值
print("统计信息摘要:")
print(df.describe())

# info(): 显示 DataFrame 的基本信息，包括非空值的数量和数据类型
print("显示DataFrame的基本信息:")
print(df.info())

# groupby(): 根据列的值对 DataFrame 进行分组
print("分组:")
"""
这段代码演示了如何使用 Pandas 中的 `groupby()` 方法对 DataFrame 进行分组，并如何获取特定分组的数据。

1. `grouped = df.groupby('City')`：首先，使用 `groupby()` 方法将 DataFrame `df` 按照 'City' 列的值进行分组。这将创建一个 GroupBy 对象 `grouped`，其中每个分组都基于 'City' 列的唯一值。

2. `grouped.get_group('New York')`：接下来，通过调用 `get_group('New York')`，你可以从分组中获取名为 'New York' 的特定分组的数据。这行代码返回一个新的 DataFrame，其中包含了 'City' 列中值为 'New York' 的所有行。

总之，这段代码将 DataFrame 按 'City' 列的唯一值分成多个分组，并且你可以使用 `get_group()` 方法获取特定分组的数据，以便进一步分析或处理该分组的数据。这对于按照某个列的值来组织和处理数据非常有用。
"""
grouped = df.groupby('City')
print(grouped.get_group('New York'))

# pivot(): 透视表操作，将数据透视为一个新的 DataFrame
pivot_df = df.pivot(index='Name', columns='City', values='Salary')
print("透视表:")
print(pivot_df)

# sort_values(): 对 DataFrame 进行排序
sorted_df = df.sort_values(by='Age', ascending=False)
print("排序:")
print(sorted_df)
print("打印values")
print(df.values)
print("分别打印出字典和df")
# 创建一个示例字典
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 24],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Chicago', 'San Francisco'],
    'Salary': [60000, 80000, 75000, 70000, 90000]
}

# 将字典转换为DataFrame
df = pd.DataFrame(data)

# 打印DataFrame
print(df)
print("打印出字典")
print(data)
