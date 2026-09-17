# 变量
name = "名称"
age = 39
pi = 3.14
is_java = True

print(type(name))
print(type(age))
print(type(pi))
print(type(is_java))    

# 字符串格式化
greeting = f"Hello, {name}! You are {age} years old."
print(greeting)

# 多行字符串
intro = """
这是一段长文本
可以换行呢
语法和java不一样
"""

print(intro)

# 列表
fruits = ["apple", "banana", "cherry"]
fruits.append("date")
fruits.append("elderberry")

print(f"水果：{fruits}")
print(f"第一个水果是：{fruits[0]}")

# 列表推导式
upper_fruits = [fruit.upper() for fruit in fruits]
print(f"大写水果：{upper_fruits}")

# 字典
person = {"name": "李四", "age": 40, "city": "北京"}
print(f"姓名：{person['name']}")
print(f"年龄：{person['age']}")
print(f"城市：{person['city']}")

# 循环
for fruit in fruits:
    print(fruit)

for key, value in person.items():
    print(f"{key}: {value}")

for i in range(5):
    print(i)

# 条件判断
if age > 30:
    print("年龄大于30")
elif age > 20:
    print("年龄大于20")
else:
    print("年龄小于等于20")

# 函数
def greet(name):
    return f"Hello, {name}!"

print(greet("王五"))





