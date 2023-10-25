# import sys：导入Python标准库中的sys模块，该模块提供了与Python解释器交互的功能，包括控制脚本的退出。
import sys
# import unittest：导入Python标准库中的unittest模块，该模块提供了用于编写和运行测试的框架。
import unittest

"""
suite = unittest.defaultTestLoader.discover('backtesting.test', pattern='_test*.py')：这行代码执行以下操作：
    unittest.defaultTestLoader 创建了一个TestLoader实例，用于加载测试用例。
    discover 方法用于发现测试用例，它接受两个参数：
    1. 'backtesting.test' 是测试用例所在的目录或包的名称。
    
    2. pattern='_test*.py' 是一个用于匹配测试文件的通配符模式。
        它表示查找以"_test"开头并以".py"结尾的Python文件，这些文件通常包含测试用例。
    discover 方法将找到的测试用例收集到一个测试套件（TestSuite）中，并将这个套件赋值给变量suite
总之，这段代码的作用是自动运行指定目录中的所有测试用例，并根据测试结果返回适当的退出码，以便在命令行中进行测试的自动化执行和集成。
"""
suite = unittest.defaultTestLoader.discover('backtesting.test',
                                            pattern='_test*.py')

"""
if __name__ == '__main__':：
    这是Python中的一种常见习惯，用于检查脚本是否作为主程序运行。
    如果脚本被直接执行（而不是被导入作为模块），则条件成立。
"""
if __name__ == '__main__':
    """
    result = unittest.TextTestRunner(verbosity=2).run(suite)：
        这行代码执行以下操作：
        创建了一个TextTestRunner的实例，它是unittest模块提供的测试运行器。
        verbosity=2 参数指定了测试运行的详细程度。
        在这里，2 表示显示详细的测试结果。
        调用run 方法来运行测试套件（suite）中的所有测试用例，并将结果存储在result变量中。
    """
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    """
    sys.exit(not result.wasSuccessful())：
        最后一行代码使用sys.exit方法退出脚本。
        result.wasSuccessful() 是一个布尔值，表示测试是否成功。
        如果所有测试用例都成功，wasSuccessful() 返回True，否则返回False。
        not result.wasSuccessful() 取反操作，将True 转换为False，反之亦然。
        如果测试成功，脚本以退出码0退出，否则以退出码1退出。
        这是一种常见的约定，其中退出码0表示成功，而退出码非零表示失败。
    """
    sys.exit(not result.wasSuccessful())
