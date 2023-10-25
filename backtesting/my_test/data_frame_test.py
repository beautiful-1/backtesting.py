import pandas as pd

# 创建一个示例 DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie', 'David'],
        'Age': [25, 30, 35, 28],
        'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']}

df = pd.DataFrame(data)

# 显示 DataFrame 的前几行
print(df.head())

# 获取 DataFrame 的维度
print(f"Shape: {df.shape}")

# 获取 DataFrame 的列名
print(f"Columns: {df.columns}")

# 选择特定列
print(df['Name'])  # 或 df.Name

# 选择多列
print(df[['Name', 'Age']])

# 过滤行
filtered_df = df[df['Age'] > 30]
print(filtered_df)

# 添加新列
df['Salary'] = [60000, 75000, 90000, 55000]
print(df)

# 删除列
df = df.drop('City', axis=1)  # axis=1 表示删除列
print(df)

# 排序 DataFrame
sorted_df = df.sort_values(by='Age', ascending=False)
print(sorted_df)

# 使用条件更新数据
df.loc[df['Name'] == 'Alice', 'Salary'] = 65000
print(df)

# 使用条件替换数据
df.loc[df['Name'] == 'Bob', 'City'] = 'Seattle'
print(df)


# 创建两个示例的索引
index1 = pd.Index(['A', 'B', 'C'])
index2 = pd.Index(['A', 'B', 'C'])
index3 = pd.Index(['A', 'B', 'D'])

# 使用 equals 方法比较索引
result1 = index1.equals(index2)  # True，因为两个索引相等
result2 = index1.equals(index3)  # False，因为两个索引不相等

print(result1)
print(result2)
