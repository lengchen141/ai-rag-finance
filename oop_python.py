# ========== 1. 基本类定义 ==========
# Java: public class Stock { private String name; ... }
# Python: 没有public/private，没有分号，没有大括号

class Stock:
    # __init__ 就是构造函数，相当于 Java 的 Stock(String name, ...)
    # self 就是 Java 的 this，但必须显式写在第一个参数
    def __init__(self, name, code, price=0):
        self.name = name        # 相当于 Java 的 this.name = name
        self.code = code
        self.price = price
        self.history = []       # 价格历史记录

    # 实例方法，第一个参数必须是 self
    def update_price(self, new_price):
        self.history.append(self.price)
        self.price = new_price

    # __str__ 相当于 Java 的 toString()
    def __str__(self):
        return f"{self.name}({self.code}) 当前价: {self.price}"

    # __repr__ 用于调试时显示，Java 没有对应物
    def __repr__(self):
        return f"Stock('{self.name}', '{self.code}', {self.price})"


# ========== 2. 继承 ==========
# Java: public class FundStock extends Stock { ... }
# Python: class FundStock(Stock):

class FundStock(Stock):
    def __init__(self, name, code, price, fund_type):
        super().__init__(name, code, price)   # 相当于 Java 的 super(name, code, price)
        self.fund_type = fund_type

    # 方法重写，和 Java 一样直接定义同名方法
    def __str__(self):
        return f"[{self.fund_type}] {self.name}({self.code}) 当前价: {self.price}"


# ========== 3. 使用 ==========
# Python 不需要 new 关键字！直接调用类名()
stock = Stock("贵州茅台", "600519", 1800)
print(stock)               # 自动调用 __str__
print(repr(stock))          # 自动调用 __repr__

stock.update_price(1820)
print(f"历史价格: {stock.history}")
print(f"当前价格: {stock.price}")

fund = FundStock("易方达蓝筹", "005827", 2.35, "混合型")
print(fund)


# ========== 4. Python 独有的特性 ==========

# 4a. 可以给实例动态添加属性（Java 做不到）
stock.market_cap = "2.26万亿"
print(f"市值: {stock.market_cap}")

# 4b. dataclass —— 相当于 Java 的 record 或 Lombok 的 @Data
from dataclasses import dataclass

@dataclass
class Order:
    stock_code: str
    quantity: int
    price: float
    direction: str = "BUY"    # 默认值

    @property   # 相当于 Java 的 getter，调用时不用加括号
    def amount(self):
        return self.quantity * self.price

order = Order("600519", 100, 1820.0)
print(f"\n订单: {order}")              # dataclass 自动生成 __str__
print(f"金额: {order.amount}")        # 像属性一样访问，实际调用了方法

# 4c. 类方法和静态方法
class Portfolio:
    _total_value = 0    # 类变量，相当于 Java 的 static field

    def __init__(self, stock, shares):
        self.stock = stock
        self.shares = shares

    @staticmethod   # 和 Java 的 static method 一样
    def market_status():
        return "A股交易时间: 9:30-15:00"

    @classmethod   # 比 Java 多了 cls 参数（类本身），可以创建工厂方法
    def from_dict(cls, data):
        stock = Stock(data["name"], data["code"], data["price"])
        return cls(stock, data.get("shares", 0))

print(f"\n{Portfolio.market_status()}")
portfolio = Portfolio.from_dict({"name": "宁德时代", "code": "300750", "price": 210, "shares": 200})
print(f"组合: {portfolio.stock}, {portfolio.shares}股")