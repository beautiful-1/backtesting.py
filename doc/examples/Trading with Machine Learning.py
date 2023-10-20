# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Trading with Machine Learning Models
#
# This tutorial will show how to train and backtest a 
# [machine learning](https://en.wikipedia.org/wiki/Machine_learning)
# price forecast model with _backtesting.py_ framework. It is assumed you're already familiar with
# [basic framework usage](https://kernc.github.io/backtesting.py/doc/examples/Quick%20Start%20User%20Guide.html)
# and machine learning in general.
#
# For this tutorial, we'll use almost a year's worth sample of hourly EUR/USD forex data:

# +
from backtesting.test import EURUSD, SMA

data = EURUSD.copy()
data

"""
机器学习概念解释

在这段代码中，`y_true` 和 `y_pred` 是用于评估机器学习模型性能的两个关键指标。下面是对它们的解释：

- `y_true`：
这是真实的目标变量或标签。在这个上下文中，`y_true` 是实际的市场价格变动方向（涨、跌、或不变）作为目标。
每个元素对应于一个时间点，表示该时间点的真实价格变动方向。

- `y_pred`：
这是模型的预测值，即机器学习模型根据输入数据（特征）预测的价格变动方向。
与 `y_true` 一样，`y_pred` 包含了对未来价格变动的预测，每个元素对应于一个时间点。

在这段代码中，`y_true` 和 `y_pred` 被用于比较模型的预测性能。
通过将它们进行比较，你可以计算不同的性能指标，如准确率、召回率、F1 分数等，以评估模型的预测能力。
通常，一个好的机器学习模型应该能够在 `y_pred` 中准确地预测与 `y_true` 相符的价格变动方向。

以下是对几种常见性能指标的解释：

- 准确率（Accuracy）：
模型正确预测的样本数占总样本数的比例，即 `(真正例 + 真负例) / 总样本数`。
在这个上下文中，它表示模型正确预测价格变动方向的比例。

- 召回率（Recall）：
模型正确预测的真正例占所有真正例的比例，即 `真正例 / (真正例 + 假负例)`。
在这个上下文中，它表示模型正确预测市场上升的比例。

- F1 分数（F1 Score）：
综合考虑了准确率和召回率，是这两者的调和平均值。
它可以用于综合评估模型的性能。

通过比较 `y_true` 和 `y_pred`，你可以计算这些性能指标，以评估模型的预测质量和准确性。
在这段代码中，`print('Classification accuracy: ', np.mean(y_test == y_pred))` 通过计算准确率来评估模型的分类性能。
"""


# -

# In
# [supervised machine learning](https://en.wikipedia.org/wiki/Supervised_learning), 
# we try to learn a function that maps input feature vectors (independent variables) into known output values (dependent variable):
#
# $$ f\colon X \to \mathbf{y} $$
#
# That way, provided our model function is sufficient, we can predict future output values from the newly acquired input feature vectors to some degree of certainty.
# In our example, we'll try to map several price-derived features and common technical indicators to the price point two days in the future.
# We construct [model design matrix](https://en.wikipedia.org/wiki/Design_matrix) $X$ below:

# +
def BBANDS(data, n_lookback, n_std):
    """Bollinger bands indicator"""
    hlc3 = (data.High + data.Low + data.Close) / 3
    mean, std = hlc3.rolling(n_lookback).mean(), hlc3.rolling(n_lookback).std()
    upper = mean + n_std * std
    lower = mean - n_std * std
    return upper, lower


close = data.Close.values
sma10 = SMA(data.Close, 10)
sma20 = SMA(data.Close, 20)
sma50 = SMA(data.Close, 50)
sma100 = SMA(data.Close, 100)
upper, lower = BBANDS(data, 20, 2)

# Design matrix / independent features:

# Price-derived features
# 特征表示了当日价格相对于其过去 10 天平均价格的偏差程度。
data['X_SMA10'] = (close - sma10) / close
data['X_SMA20'] = (close - sma20) / close
data['X_SMA50'] = (close - sma50) / close
data['X_SMA100'] = (close - sma100) / close


"""
这个特征表示了10日和20日移动平均线之间的差异程度，可用于捕捉价格趋势的快速变化。
data['X_DELTA_SMA20'] = (sma20 - sma50) / close: 
    类似于第一行，这一行创建了一个名为 'X_DELTA_SMA20' 的特征列，
    其中包含了20日简单移动平均线 sma20 与50日简单移动平均线 sma50 之间的差异相对于 close 的比例。
    这个特征表示了20日和50日移动平均线之间的
"""
data['X_DELTA_SMA10'] = (sma10 - sma20) / close
data['X_DELTA_SMA20'] = (sma20 - sma50) / close
data['X_DELTA_SMA50'] = (sma50 - sma100) / close

# Indicator features
"""
data['X_MOM'] = data.Close.pct_change(periods=2): 这一行创建了一个名为 'X_MOM' 的特征列，其中包含了股票的两天累积收益率。
data.Close.pct_change(periods=2) 计算了股价的两天百分比变化，表示了过去两天的价格变化情况。
"""
data['X_MOM'] = data.Close.pct_change(periods=2)

"""
data['X_BB_upper'] = (upper - close) / close: 这一行创建了一个名为 'X_BB_upper' 的特征列，
其中包含了股价与Bollinger Bands（布林带）上轨之间的差异相对于 close 的比例。
Bollinger Bands 是一种技术指标，用于度量价格的波动性，
这个特征表示价格相对于上轨的位置。
"""
data['X_BB_upper'] = (upper - close) / close
"""
data['X_BB_lower'] = (lower - close) / close: 类似于第二行，这一行创建了一个名为 'X_BB_lower' 的特征列，
其中包含了股价与Bollinger Bands下轨之间的差异相对于 close 的比例。
这个特征表示价格相对于下轨的位置。
"""
data['X_BB_lower'] = (lower - close) / close

"""
data['X_BB_width'] = (upper - lower) / close: 这一行创建了一个名为 'X_BB_width' 的特征列，
其中包含了Bollinger Bands的带宽（上轨和下轨之间的距离）相对于 close 的比例。
这个特征可以用于度量价格波动性的变化。
"""
data['X_BB_width'] = (upper - lower) / close

"""
data['X_Sentiment'] = ~data.index.to_series().between('2017-09-27', '2017-12-14'): 最后一行创建了一个名为 'X_Sentiment' 的特征列，
其中包含了布尔值，表示股价数据的日期是否在指定的日期范围内。
这个特征用于捕捉可能与某些时间段相关的市场情绪或事件。
"""
data['X_Sentiment'] = ~data.index.to_series().between('2017-09-27', '2017-12-14')

# Some datetime features for good measure
"""
data['X_day'] = data.index.dayofweek: 这一行代码创建了一个名为 'X_day' 的特征列，
其中包含了数据中每个时间戳所对应的星期几。
data.index 包含了时间戳信息，dayofweek 方法返回一个整数，表示星期几，其中0表示星期一，1表示星期二，以此类推，6表示星期日。
这个特征可以用于捕捉周内的季节性变化或市场交易日的不同表现
"""
data['X_day'] = data.index.dayofweek

"""
data['X_hour'] = data.index.hour: 这一行代码创建了一个名为 'X_hour' 的特征列，其中包含了数据中每个时间戳所对应的小时数。
data.index 中的 hour 属性返回一个整数，表示每个时间戳的小时部分（24小时制）。
这个特征可以用于捕捉每天不同时间段内市场的行为变化，例如开盘和收盘时段。
"""
data['X_hour'] = data.index.hour

"""
data.dropna(): 这是一个数据清理操作，它用于删除数据中包含空值（NaN）的行。
在Pandas中，NaN表示缺失的数据或不可用的数据。通过使用dropna()方法，可以轻松删除包含NaN值的行，以确保数据是干净的，没有缺失值。

.astype(float): 这是一个数据类型转换操作。
它将数据框中的所有列的数据类型更改为浮点数类型。
通常，数据帧（DataFrame）中的数据可以具有不同的数据类型，例如整数、字符串、浮点数等。
在这种情况下，它确保了整个数据帧中的数据都以浮点数的形式存储，这对于后续的计算和分析可能是有用的。
"""
data = data.dropna().astype(float)
# -

# Since all our indicators work only with past values, we can safely precompute the design matrix in advance. Alternatively, we would reconstruct the matrix every time before training the model.
#
# Notice the made-up _sentiment_ feature. In real life, one would obtain similar features by parsing news sources, Twitter sentiment, Stocktwits or similar.
# This is just to show input data can contain all sorts of additional explanatory columns.
#
# As mentioned, our dependent variable will be the price (return) two days in the future, simplified into values $1$ when the return is positive (and significant), $-1$ when negative, or $0$ when the return after two days is roughly around zero. Let's write some functions that return our model matrix $X$ and dependent, class variable $\mathbf{y}$ as plain NumPy arrays:

# +
import numpy as np

"""
data.filter(like='X')：这部分代码使用 Pandas 的 filter 方法来选择数据框中列名（特征名称）中包含字符串 'X' 的所有列。
在机器学习中，特征通常以 'X' 开头，因此这行代码的目的是选择所有特征列。
结果将是一个子数据框，其中只包含以 'X' 开头的列。

.values：这部分代码将选择的子数据框转换为一个 NumPy 数组。
在机器学习中，通常需要将特征数据表示为 NumPy 数组，以便用于模型训练。
"""


def get_X(data):
    """Return model design matrix X"""
    return data.filter(like='X').values


"""
这是一个函数，用于从给定的数据框 `data` 中计算并返回依赖变量 `y`。在这种情况下，`y` 代表了一个二元分类目标变量，用于机器学习模型。

- `data.Close.pct_change(48)`：
这部分代码计算了价格数据的百分比变化，具体来说是收盘价格在 48 个时间步长内的百分比变化。
这相当于计算了大约两天后的价格相对于当前价格的变化率。
结果是一个包含价格变化率的数据序列。

- `.shift(-48)`：
这部分代码将上述计算的百分比变化数据序列向前移动了 48 个时间步长。
这个操作的目的是将这些变化率与相应的未来价格变化相关联，因为我们的目标是预测大约两天后的价格变化。

- `y[y.between(-.004, .004)] = 0`：
这行代码将 `y` 数据中的绝对值小于 0.4% 的价格变化率设置为 0。
这是为了将价格变化率接近零的情况归类为不显著的价格变化，以减少噪音对模型的影响。

- `y[y > 0] = 1` 和 `y[y < 0] = -1`：
这部分代码将 `y` 数据中的正价格变化率（涨幅）设置为 1，'而负价格变化率（跌幅）设置为 -1。
这是为了将问题转化为一个二元分类问题，其中正价格变化表示一个类别，负价格变化表示另一个类别。

最终，该函数返回一个代表目标变量 `y` 的 Pandas 数据序列，其中包含了 1（正价格变化）和 -1（负价格变化）作为类别标签。
这个目标变量将用于机器学习模型的训练和预测。
"""


def get_y(data):
    """Return dependent variable y"""
    y = data.Close.pct_change(48).shift(-48)  # Returns after roughly two days
    y[y.between(-.004, .004)] = 0  # Devalue returns smaller than 0.4%
    y[y > 0] = 1
    y[y < 0] = -1
    return y


"""
这是一个函数，用于从给定的数据框 `df` 中获取独立特征变量 `X` 和依赖变量 `y`，并确保数据中不包含 NaN（缺失）值。函数执行以下步骤：

1. `X = get_X(df)`：
调用另一个函数 `get_X` 来获取特征矩阵 `X`，它包含了数据框 `df` 中以 "X" 开头的列，这些列是用于特征工程的独立特征。

2. `y = get_y(df).values`：
调用另一个函数 `get_y` 来获取依赖变量 `y`，这是用于机器学习模型的目标变量。
`.values` 将 Pandas 数据序列转换为 NumPy 数组。

3. `isnan = np.isnan(y)`：
创建一个布尔数组 `isnan`，其中的元素对应于 `y` 中的 NaN 值。
这将帮助识别 `y` 中的缺失值。

4. `X = X[~isnan]` 和 `y = y[~isnan]`：
使用 `isnan` 数组，将 `X` 和 `y` 中对应于 NaN 值的行从数据中删除，从而确保 `X` 和 `y` 不包含任何 NaN 值。

5. 最后，函数返回清理后的特征矩阵 `X` 和目标变量 `y`，它们已经准备好用于机器学习模型的训练和预测。
"""


def get_clean_Xy(df):
    """Return (X, y) cleaned of NaN values"""
    X = get_X(df)
    y = get_y(df).values
    isnan = np.isnan(y)
    print(isnan)
    X = X[~isnan]
    y = y[~isnan]
    return X, y


# -

# Let's see how our data performs modeled using a simple
# [k-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
# (kNN) algorithm from the state of the art
# [scikit-learn](https://scikit-learn.org)
# Python machine learning package.
# To avoid (or at least demonstrate)
# [overfitting](https://scikit-learn.org/stable/auto_examples/model_selection/plot_underfitting_overfitting.html),
# always split your data into _train_ and _test_ sets; in particular, don't validate your model performance on the same data it was built on.

# +
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

"""
这段代码执行以下操作：

1. `X, y = get_clean_Xy(data)`：
调用之前定义的 `get_clean_Xy` 函数，从数据框 `data` 中获取清理后的特征矩阵 `X` 和目标变量 `y`。
这些数据准备用于机器学习模型的训练和测试。

2. `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)`：
使用 `train_test_split` 函数将数据 `X` 和 `y` 分割成训练集（`X_train` 和 `y_train`）和测试集（`X_test` 和 `y_test`）。
`test_size=.5` 表示将数据的一半用于测试，`random_state=0` 用于设置随机种子，以确保分割是可重现的。

3. `clf = KNeighborsClassifier(7)`：
创建一个 k-最近邻（k-Nearest Neighbors）分类器，其中 `7` 表示模型会考虑 7 个最近的邻居来进行分类。

4. `clf.fit(X_train, y_train)`：
使用训练集 `X_train` 和对应的目标变量 `y_train` 训练 k-最近邻分类器。
模型将学习如何根据特征向量 `X` 来预测对应的目标变量 `y`。

5. `y_pred = clf.predict(X_test)`：
使用训练好的模型 `clf` 针对测试集 `X_test` 进行预测，得到预测结果 `y_pred`，这是模型对测试集中每个样本的分类结果。

6. `_ = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).plot(figsize=(15, 2), alpha=.7)`：
创建一个 Pandas 数据框，其中包含真实目标变量 `y_true` 和模型的预测结果 `y_pred`。
然后，使用 `plot` 函数将这些结果可视化。`figsize=(15, 2)` 用于设置图形的尺寸。

7. `print('Classification accuracy: ', np.mean(y_test == y_pred))`：
计算分类准确度，即模型正确分类的样本比例，然后将结果打印出来。 
`np.mean(y_test == y_pred)` 表示比较 `y_test` 和 `y_pred`，并计算相等的比例，这对于了解模型的性能非常有用。
"""
X, y = get_clean_Xy(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5, random_state=0)

clf = KNeighborsClassifier(7)  # Model the output based on 7 "nearest" examples

clf.fit(X_train, y_train)
"""
这段代码执行了一个机器学习模型的预测操作。让我逐步解释它：

1. `clf` 是一个机器学习模型，通常是一个已经训练好的分类器。
    在这种情况下，`clf` 是一个K近邻分类器，它在训练过程中学习了如何根据输入数据来对其进行分类。

2. `X_test` 是测试数据集，包含了模型在训练过程中没有见过的输入特征数据。
    这些数据通常用于评估模型在未知数据上的性能。

3. `clf.predict(X_test)` 是一个模型预测的操作。
    模型使用测试数据 `X_test` 中的特征数据来进行预测，并将预测结果存储在 `y_pred` 变量中。

4. `y_pred` 是一个包含了模型对测试数据的预测结果的数组。
    如果模型是一个分类器，那么 `y_pred` 中的每个元素将是对应测试数据点的类别标签。
    如果模型是一个回归器，那么 `y_pred` 中的每个元素将是对应测试数据点的数值预测。

这个过程的目的是评估模型在未知数据上的性能，通常通过与测试数据的真实标签或目标值进行比较，以计算模型的准确性或其他性能指标。
这些预测结果可以用于后续的分析、可视化或决策制定。
"""
y_pred = clf.predict(X_test)
"""
这段代码执行了以下操作：

1. `pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})`：
    首先创建了一个Pandas DataFrame对象，其中包含了两列数据。`'y_true'` 包含了测试数据的真实目标值（或标签），`'y_pred'` 包含了模型对相应测试数据的预测值。

2. `.plot(figsize=(15, 2), alpha=.7)`：
    接下来，对上述的DataFrame对象进行绘图操作。在这里，`plot` 是一个Pandas DataFrame对象的方法，用于可视化数据。

   - `figsize=(15, 2)` 设置了图形的大小，指定宽度为15单位，高度为2单位。
   - `alpha=.7` 设置了绘图中元素的透明度，这里为0.7，使得图中的数据点稍微透明，以便更好地观察重叠的点。

整体来说，这段代码的目的是创建一个包含真实目标值和模型预测值的数据帧，并绘制这两列数据的折线图，以便比较它们。
这可以帮助你直观地了解模型的性能，查看真实值和预测值之间的关系，以便进行进一步的分析和评估。
"""
_ = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).plot(figsize=(15, 2), alpha=.7)

"""
这行代码用于计算并打印分类准确度（Classification accuracy）的值。让我解释一下每个部分：

- `'Classification accuracy: '`：这是一个简单的文本字符串，用于打印在屏幕上，作为输出的前缀。

- `np.mean(y_test == y_pred)`：这是实际的计算部分。
  - `y_test`是测试数据的实际标签（真实值）。
  - `y_pred`是模型预测的标签。
  - `y_test == y_pred`执行逐元素的比较，对于每个元素，如果预测与实际值匹配，它将返回`True`，否则返回`False`。这将生成一个布尔数组。

  - `np.mean(...)`计算布尔数组中`True`值的比例，因为`True`在数值上等同于1，`False`等同于0。这样，`np.mean(y_test == y_pred)`将返回分类准确度的比例，即模型正确分类的样本占总样本数的比例。

所以，`print('Classification accuracy: ', np.mean(y_test == y_pred))`的输出将是包括前缀文本和分类准确度值的字符串，例如：

```
Classification accuracy: 0.85
``` 

这表示模型的分类准确度为85％，即模型正确分类的样本占总样本数的85％。
"""
print('Classification accuracy: ', np.mean(y_test == y_pred))


# -

# We see the forecasts are all over the place (classification accuracy 42%), but is the model of any use under real backtesting?
#
# Let's backtest a simple strategy that buys the asset for 20% of available equity with 20:1 leverage whenever the forecast is positive (the price in two days is predicted to go up),
# and sells under the same terms when the forecast is negative, all the while setting reasonable stop-loss and take-profit levels. Notice also the steady use of
# [`data.df`](https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.data)
# accessor:

# +
# %%time

from backtesting import Backtest, Strategy

"""
N_TRAIN = 400：这行代码定义了一个名为 N_TRAIN 的常量，它的值为 400。
这个常量通常用于指定在策略中使用多少个数据点来进行训练，特别是在机器学习模型的训练中。
在这个上下文中，N_TRAIN 可能表示在策略的初始训练期间使用的历史数据点数量。
"""
N_TRAIN = 400

"""
这个策略类的初始化方法主要完成了以下操作：创建机器学习模型、使用历史数据训练模型、创建指标来存储真实价格方向和模型的价格预测。在策略的后续执行中，这个模型将被用于价格预测和决策制定。
"""


class MLTrainOnceStrategy(Strategy):
    # 这一行定义了一个类属性 price_delta，它的值为 0.004，表示价格变化的阈值，即0.4%。
    price_delta = .004  # 0.4%

    def init(self):
        # Init our model, a kNN classifier
        # 在初始化方法中创建了一个 k-最近邻分类器（k-Nearest Neighbors），并将其存储在 self.clf 中。
        # 这个分类器用于进行价格的预测。
        self.clf = KNeighborsClassifier(7)

        # Train the classifier in advance on the first N_TRAIN examples
        # 从策略的 self.data 属性中获取历史数据，并使用 .iloc[:N_TRAIN] 选择前 N_TRAIN 行数据。
        # 这是为了用前 N_TRAIN 行数据来训练机器学习模型。
        df = self.data.df.iloc[:N_TRAIN]
        # 调用 get_clean_Xy 函数，将选定的历史数据 df 转换为特征矩阵 X 和目标变量 y，然后将它们存储在 X 和 y 变量中。
        X, y = get_clean_Xy(df)

        # 使用前面初始化的 k-最近邻分类器 self.clf，对特征矩阵 X 和目标变量 y 进行训练，以构建价格预测模型。
        self.clf.fit(X, y)

        # Plot y for inspection
        # 这一行通过 self.I 方法创建一个指标，该指标使用 get_y 函数从 self.data.df 中提取真实的目标变量，然后命名为 'y_true'。
        # 这个指标用于将真实的价格方向可视化，以便进行检查。
        self.I(get_y, self.data.df, name='y_true')

        # Prepare empty, all-NaN forecast indicator
        # 这一行创建一个指标 self.forecasts，其值为全是 NaN 值（未知），长度与历史数据的长度相同，然后指标的名称为 'forecast'。
        # 这个指标用于存储模型的价格预测结果，开始时是空的。

        """
        这段代码是一个`lambda`表达式，用于创建一个NumPy数组，其中所有的元素都是`NaN`（不是一个数字）。让我们逐步解释它：
        1. `lambda:`：这是Python中用于创建匿名函数（无需为函数指定名称）的关键字。在这里，`lambda`关键字引导了一个匿名函数的定义。
        2. `np.repeat(np.nan, len(self.data))`：这是匿名函数的主体部分。
            它使用NumPy库中的`np.repeat`函数来生成一个包含`NaN`的数组。
            具体解释如下：
            - `np.nan`：`
                np.nan`表示“不是一个数字”，通常用于表示缺失值或未初始化的数据。
            - `len(self.data)`：
                `len(self.data)`计算了`self.data`的长度，即数据中的数据点数目。
            - `np.repeat(a, repeats)`：
                这个NumPy函数会复制数组`a`中的元素，重复`repeats`次，然后将它们连接在一起。
                在这里，`a`是一个包含单个`NaN`元素的数组，`repeats`是数据点的数量。
                这就意味着它将重复`NaN`元素，使其与数据点数量相匹配。
                
        所以，这段代码的目的是创建一个与`self.data`的长度相同的NumPy数组，该数组的所有元素都是`NaN`。
        这通常用于初始化一个指标或数组，以便在后续的计算中填充和更新数据。
        在这个上下文中，`self.forecasts`将用这个数组初始化，然后在策略的执行中进行更新以存储预测值。
        """
        self.forecasts = self.I(lambda: np.repeat(np.nan, len(self.data)), name='forecast')

    def next(self):
        # Skip the training, in-sample data
        # 这一行代码检查当前数据点的数量是否小于 N_TRAIN，也就是策略的初始训练期。
        # 如果是，就直接返回，跳过策略逻辑的执行，因为在初始训练期内不执行交易。
        if len(self.data) < N_TRAIN:
            return

        # Proceed only with out-of-sample data. Prepare some variables

        # 这一行代码从 self.data 中获取当前时间步的高、低和收盘价格，以备后续使用。
        high, low, close = self.data.High, self.data.Low, self.data.Close
        # 这一行代码获取当前时间步的时间戳，通常是时间序列数据中的日期和时间。
        current_time = self.data.index[-1]

        # Forecast the next movement
        # 这一行代码构建模型输入特征 X，通过调用 get_X 函数，它获取了最新的历史数据点，并转化为模型需要的特征格式。
        X = get_X(self.data.df.iloc[-1:])

        # 这一行代码使用事先训练好的机器学习模型 clf 对输入特征 X 进行预测，得到 forecast，即模型对未来市场价格走势的预测。
        forecast = self.clf.predict(X)[0]

        # Update the plotted "forecast" indicator
        # 一行代码将最新的预测结果 forecast 存储在名为 self.forecasts 的指标中，以供后续分析和可视化使用。
        self.forecasts[-1] = forecast

        # If our forecast is upwards and we don't already hold a long position
        # place a long order for 20% of available account equity. Vice versa for short.
        # Also set target take-profit and stop-loss prices to be one price_delta
        # away from the current closing price.

        # 接下来的一段代码用于根据预测结果执行交易决策：
        #
        # 如果 forecast 等于 1，并且当前没有持有多头头寸 (self.position.is_long 为假)，则执行买入操作。
        #   这里规定买入的头寸大小为总账户资产的 20%（size=.2），并设置止盈（tp）和止损（sl）价格。
        # 如果 forecast 等于 -1，并且当前没有持有空头头寸 (self.position.is_short 为假)，
        #   则执行卖出操作，同样规定卖出的头寸大小为总账户资产的 20%。
        upper, lower = close[-1] * (1 + np.r_[1, -1] * self.price_delta)

        if forecast == 1 and not self.position.is_long:
            self.buy(size=.2, tp=upper, sl=lower)
        elif forecast == -1 and not self.position.is_short:
            self.sell(size=.2, tp=lower, sl=upper)

        # Additionally, set aggressive stop-loss on trades that have been open 
        # for more than two days
        # 最后的一段代码用于设置交易的激进止损。
        # 如果某笔交易已经开仓超过两天（current_time - trade.entry_time > pd.Timedelta('2 days')），
        # 则会将止损价（trade.sl）调整为较高的值（对于多头头寸）或较低的值（对于空头头寸），以限制潜在的亏损。
        for trade in self.trades:
            if current_time - trade.entry_time > pd.Timedelta('2 days'):
                if trade.is_long:
                    trade.sl = max(trade.sl, low)
                else:
                    trade.sl = min(trade.sl, high)


bt = Backtest(data, MLTrainOnceStrategy, commission=.0002, margin=.05)
bt.run()
# -

bt.plot()

# Despite our lousy win rate, the strategy seems profitable. Let's see how it performs under
# [walk-forward optimization](https://en.wikipedia.org/wiki/Walk_forward_optimization),
# akin to k-fold or leave-one-out
# [cross-validation](https://en.wikipedia.org/wiki/Cross-validation_%28statistics%29):

# +
# %%time
"""
综合起来，MLWalkForwardStrategy 是一种滚动训练和回测策略，它在每个时间步中检查是否需要重新训练模型，以确保模型能够适应最新的市场数据。
这有助于提高模型的鲁棒性和适应性。
"""


class MLWalkForwardStrategy(MLTrainOnceStrategy):
    def next(self):
        # Skip the cold start period with too few values available
        if len(self.data) < N_TRAIN:
            return

        # Re-train the model only every 20 iterations.
        # Since 20 << N_TRAIN, we don't lose much in terms of
        # "recent training examples", but the speed-up is significant!
        """
        这一行代码检查当前数据点的数量是否可以被 20 整除。如果不能整除，说明不是进行重新训练的时机，因此直接返回，不执行重新训练。
        这是为了避免频繁重新训练模型，从而节省时间。
        """
        if len(self.data) % 20:
            return super().next()

        # Retrain on last N_TRAIN values
        """
        这一行代码从数据中选择最近的 N_TRAIN 个数据点，用于重新训练模型。
        这是一种滚动窗口的方式，确保我们一直在使用最新的数据来训练模型。
        """
        df = self.data.df[-N_TRAIN:]
        """
        这一行代码用于获取特征矩阵 X 和目标变量 y，它们将被用于模型的重新训练。
        get_clean_Xy 函数从给定的数据中提取特征和目标变量，并移除其中的 NaN 值。
        """
        X, y = get_clean_Xy(df)
        """
        这一行代码重新训练机器学习模型 clf，使用最新的特征矩阵 X 和目标变量 y。这样，模型将适应最新的市场情况。
        """
        self.clf.fit(X, y)

        # Now that the model is fitted, 
        # proceed the same as in MLTrainOnceStrategy
        """
        这一行代码调用父类 MLTrainOnceStrategy 中的 next 方法，执行与模型预测和交易相关的逻辑。
        因为模型已经重新训练，所以这个逻辑会使用最新的模型来进行交易决策。
        """
        super().next()


bt = Backtest(data, MLWalkForwardStrategy, commission=.0002, margin=.05)
bt.run()
# -

bt.plot()

# Apparently, when repeatedly retrained on past `N_TRAIN` data points in a rolling manner, our basic model generalizes poorly and performs not quite as well.
#
# This was a simple and contrived, tongue-in-cheek example that shows one way to use machine learning forecast models with _backtesting.py_ framework.
# In reality, you will need a far better feature space, better models (cf.
# [deep learning](https://en.wikipedia.org/wiki/Deep_learning#Deep_neural_networks)),
# and better money management strategies to achieve
# [consistent profits](https://en.wikipedia.org/wiki/Day_trading#Profitability)
# in automated short-term forex trading. More proper data science is an exercise for the keen reader.
#
# Some instant optimization tips that come to mind are:
# * **Data is king.** Make sure your design matrix features as best as possible model and correlate with your chosen target variable(s) and not just represent random noise.
# * Instead of modelling a single target variable $y$, model a multitude of target/class variables, possibly better designed than our "48-hour returns" above.
# * **Model everything:** forecast price, volume, time before it "takes off", SL/TP levels,
#   [optimal position size](https://en.wikipedia.org/wiki/Kelly_criterion#Application_to_the_stock_market)
#   ...
# * Reduce
#   [false positives](https://en.wikipedia.org/wiki/False_positive_rate)
#   by increasing the conviction needed and imposing extra domain expertise and discretionary limitations before entering trades.
#
# Also make sure to familiarize yourself with the full
# [Backtesting.py API reference](https://kernc.github.io/backtesting.py/doc/backtesting/index.html#header-submodules)
