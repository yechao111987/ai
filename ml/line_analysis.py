from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 加载数据
diabetes = datasets.load_diabetes()
# 得到患者数据
data = diabetes.data

# 数据探索
print(data.shape)
print(data[0])

# 训练集 70%，测试集30%
train_x, test_x, train_y, test_y = train_test_split(diabetes.data, diabetes.target, test_size=0.3, random_state=14)
print(len(train_x))
# 回归训练及预测
# fit 函数是自动对X Y进行线性拟合
clf = linear_model.LinearRegression()
clf.fit(train_x, train_y)
# coef_得到函数的系数 intercept_  函数截距  score 拟合分数
print(clf.coef_)
# print(train_x.shape)
# print(clf.score(test_x, test_y))
pred_y = clf.predict(test_x)
# MSE均方误差
print(mean_squared_error(test_y, pred_y))
