# 创建一个示例类
class ExampleClass:
    def __init__(self):
        self.attribute1 = 42

    def method1(self):
        pass

    def method2(self):
        pass

# 创建一个示例对象

example_object = ExampleClass()

# 列出示例对象的属性和方法
attributes_and_methods = dir(example_object)

# 打印属性和方法的列表
print(attributes_and_methods)
