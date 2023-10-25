from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource
import pandas as pd
from bokeh.transform import factor_cmap


#trades_cmap = factor_cmap('returns_positive', colors_darker, ['0', '1'])


def _plot_ohlc_trades():
    """Trade entry / exit markers on OHLC plot"""
    trade_source.add(trades[['EntryBar', 'ExitBar']].values.tolist(), 'position_lines_xs')
    trade_source.add(trades[['EntryPrice', 'ExitPrice']].values.tolist(), 'position_lines_ys')
    fig_ohlc.multi_line(xs='position_lines_xs', ys='position_lines_ys',
                        source=trade_source,
                        legend_label=f'Trades ({len(trades)})',
                        line_width=8, line_alpha=1, line_dash='dotted')


# 示例交易数据，这是一个包含进入和退出点信息的 Pandas DataFrame
trades = pd.DataFrame({
    'EntryBar': [5, 10, 15],  # 示例进入点的位置
    'ExitBar': [8, 12, 18],  # 示例退出点的位置
    'EntryPrice': [100, 105, 98],  # 示例进入价格
    'ExitPrice': [110, 102, 105]  # 示例退出价格
})

# 创建一个 Bokeh 图表
fig_ohlc = figure()

# 创建一个 Bokeh 数据源对象
trade_source = ColumnDataSource()

# 调用 _plot_ohlc_trades 函数将交易数据添加到图表中
_plot_ohlc_trades()

# 显示图表
show(fig_ohlc)
