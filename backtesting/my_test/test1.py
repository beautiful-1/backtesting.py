import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
from bokeh.io import output_notebook

# 示例数据（假设有K线和MACD数据）
# 你需要将下面的数据替换为你自己的实际数据
kline_data = pd.DataFrame({
    'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'Open': [100, 105, 110, 108, 112],
    'High': [105, 112, 115, 112, 115],
    'Low': [98, 102, 107, 105, 110],
    'Close': [103, 110, 113, 109, 114],
    'Volume': [10000, 12000, 14000, 11000, 13000]
})


# 创建K线图
def create_candlestick_chart(df):
    source = ColumnDataSource(df)
    p = figure(x_axis_type="datetime", title="K线图", width=800, height=800)
    p.segment(x0='Date', y0='Low', x1='Date', y1='High', source=source, line_color="black")
    p.vbar('index', 0.5, 'Open', 'Close', source=source, fill_color="green", line_color="black")
    return p


# 创建K线图
kline_chart = create_candlestick_chart(kline_data)

# 显示K线图
show(kline_chart)
