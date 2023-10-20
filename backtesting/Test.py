import pandas as pd
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot
from bokeh.io import output_notebook

# 初始化Bokeh的Notebook输出（适用于Jupyter Notebook）
# output_notebook()

# 示例K线数据（你需要将下面的数据替换为你自己的实际数据）
kline_data = pd.DataFrame({
    'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
    'open': [100, 105, 110, 108, 112],
    'high': [105, 112, 115, 112, 115],
    'low': [98, 102, 107, 105, 110],
    'close': [103, 90, 113, 109, 114],
    'volume': [10000, 12000, 14000, 11000, 13000]
})


# 创建K线图
def create_candlestick_chart(df):
    source = ColumnDataSource(df)

    inc = df.close >= df.open
    dec = df.open > df.close

    inc_source = ColumnDataSource(df.loc[inc])
    dec_source = ColumnDataSource(df.loc[dec])
    p = figure(x_axis_type="datetime", title="K线图", width=1000, height=800)

    hover = HoverTool(tooltips=[('date', '@date'),
                                ('open', '@open'),
                                ('high', '@high'),
                                ('low', '@low'),
                                ('close', '@close'),
                                ('pct_change', "@pct_change")
                                ]
                      )

    p.segment(x0='index', y0='high', x1='index', y1='low', color='red', source=inc_source)
    p.segment(x0='index', y0='high', x1='index', y1='low', color='green', source=dec_source)
    p.vbar('index', 0.5, 'open', 'close', fill_color='red', line_color='red', source=inc_source)
    p.vbar('index', 0.5, 'open', 'close', fill_color='green', line_color='green', source=dec_source,
           )

    # add hover tool
    p.add_tools(hover)
    return p


# 创建K线图
kline_chart = create_candlestick_chart(kline_data)

# 显示K线图
show(kline_chart)
