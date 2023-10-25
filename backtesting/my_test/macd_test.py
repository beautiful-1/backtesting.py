from bokeh.plotting import figure, show
import pandas as pd
import numpy as np

# 创建示例数据（确保长度相同）
dates = pd.date_range(start='2023-01-01', end='2023-02-31')
data_length = len(dates)
macd_values = np.random.randn(data_length)
signal_values = np.random.randn(data_length)
histogram_values = np.random.randn(data_length)  # 确保与其他数组长度相同

# 创建Pandas DataFrame
df = pd.DataFrame({
    'date': dates,
    'macd': macd_values,
    'signal': signal_values,
    'histogram': histogram_values
})

# 创建Bokeh图表
# output_notebook()

p = figure(x_axis_type="datetime", title="MACD Indicator")

# 添加MACD和Signal曲线
p.line(df['date'], df['macd'], line_width=2, line_color="blue", legend_label="MACD")
p.line(df['date'], df['signal'], line_width=2, line_color="red", legend_label="Signal")

# 添加Histogram柱状图
p.vbar(x=df['date'], bottom=0, top=df['histogram'], width=0.4, color="green", legend_label="Histogram")

# 设置图例位置
p.legend.location = "top_left"

# 显示图表
show(p)
