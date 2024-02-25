# Adidas-Us-retailers-demand-forcasting-
Анализ и предсказание продаж компании Adidas на основе ограниченного количества признаков. 
```
#Чистка Данных
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("Adidas_Another version.csv", sep=";")
data = df
df.head(10)

df = data
print(df.columns)
df = df.drop(columns=['Retailer ID', 'Units Sold', 'Operating Profit', 'Operating Margin'], axis=1)

# Оставляем retailer, date, region, product, total sales (прогнозируем), sales method
df = df.drop(columns=['State', 'City'])
df = df.drop('Unnamed: 13', axis=1)
df.head(10)

mask_dollar = lambda x: float(str(x).replace("$", "").replace(u"\xa0", "").replace(",", ".").replace(" ", ""))
df["Total Sales"] = df["Total Sales"].apply(mask_dollar)
df["Price per Unit"] = df["Price per Unit"].apply(mask_dollar)
df.head(10)

df =df.drop("Invoice Date", axis=1)

data_forgrap = df.copy()

df = pd.get_dummies(df, columns=["Region", "Retailer", "Sales Method", "Product"], dtype=int)
# for column in ["Region", "Retailer", "Sales Method", "Product"]:
    #df[column] = pd.factorize(df[column])[0]
df.head(10)

df2 = df.copy()

#df2 = df2.drop("Price per Unit", axis=1)

df2

df2 =df2.drop([9651], axis=0)
data_forgrap =data_forgrap.drop([9651], axis=0)

df2 =df2.drop([9650], axis=0)
data_forgrap =data_forgrap.drop([9650], axis=0)

df2 =df2.drop([9649], axis=0)
data_forgrap =data_forgrap.drop([9649], axis=0)

df2 =df2.drop([9648], axis=0)
data_forgrap =data_forgrap.drop([9648], axis=0)

df2.isna().sum()

df2 = df2.sample(frac=1).reset_index(drop=True)

# ГРАФИКИ ИЗНАЧАЛЬНЫХ ДАННЫХ

data_forgrap

data_forgrap['Price per Unit'].plot(kind='hist',bins=25,figsize=(12,7),title='Products price distribution',color="#FB8500")

#top_products= data_forgrap[['Price per Unit','Total Sales']].groupby('Price per Unit').mean()
#top_products.head(10)
#top_products.plot(kind='bar', figsize=(12,7),title='Total sales per Price', color="#FB8500")
#plt.show()
#top_products= data_forgrap[['Price per Unit','Total Sales']].groupby('Price per Unit')
#top_products.head(10)
#top_products.describe()

top_products = data_forgrap.groupby(pd.cut(data_forgrap['Price per Unit'], np.arange(0, 101, 10)))["Total Sales"].mean()
top_products.plot(kind='bar', figsize=(12,7),title='Total sales per Price', color="#FB8500")
plt.show()


top_products= data_forgrap[['Product','Total Sales']].groupby('Product').mean()
top_products.plot(kind='bar',figsize=(12,7),title='Mean sales per Product', color="#FB8500")
plt.show()

top_salesmt= data_forgrap[['Sales Method','Total Sales']].groupby('Sales Method').mean()
top_salesmt.plot(kind='barh',title='Top Selling Methods', color="#FB8500")

top_retailers= data_forgrap[['Retailer','Total Sales']].groupby('Retailer').mean()

#top_retailers.plot(kind='pie',subplots=True,figsize=(12,7),title='Total sales per Retailer',autopct='%1.0f%%')

sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.pie(top_retailers["Total Sales"], labels = top_retailers.index, colors = sns.color_palette("flare"), autopct='%.0f%%')
plt.show()


plt.figure(figsize=(15,15))
corr = df2.corr()
sns.heatmap(corr, cmap=sns.color_palette("flare"), vmin=-1, center=0, square=True)

import plotly.express as px
fig = px.histogram(df2, x="Price per Unit", title="Distribution of PPU")
fig.show()

sns.boxplot(df2["Total Sales"])
sns.set(rc={'figure.figsize':(17, 17)})

# Линейная Регрессия на ненормализированных данных

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


df2

Xlin = df2.values[:,(0,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]
Ylin = df2.values[:, 1]


Xlin_train, Xlin_test, ylin_train, ylin_test = train_test_split(Xlin, Ylin, test_size=0.1, random_state=0)

lr = LinearRegression()
lr.fit(Xlin_train,ylin_train)
ylin_pred = lr.predict(Xlin_test)



# Рассчет MSE
mse = np.mean((ylin_test - ylin_pred)**2)

# Рассчет MAPE
mape = np.mean(np.abs((ylin_test - ylin_pred) / ylin_test)) * 100

print("MSE:", mse)
print("MAPE:", mape)
#89.2313454435439

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

x, y = np.random.random((2, 100))*2
fig, ax = plt.subplots()
ax.scatter(ylin_pred, ylin_test)

plt.show()

# Cat BOOST на ненормализированных данных

Xcat = df2.values[:,(0,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]
Ycat = df2.values[:, 1]


Xcat_train, Xcat_test, ycat_train, ycat_test = train_test_split(Xcat, Ycat, test_size=0.1, random_state=0)

from catboost import Pool, CatBoostRegressor

# initialize Pool
train_pool = Pool(Xcat_train, ycat_train)
test_pool = Pool(Xcat_test, ycat_test)

# specify the training parameters
model = CatBoostRegressor(iterations=2500, loss_function='MAE', verbose=100)
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
predscat = model.predict(test_pool)

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(ycat_test, predscat, squared=False)
mae = mean_absolute_error(ycat_test, predscat)
# Рассчет MAPE
mape = np.mean(np.abs((ycat_test - predscat) / ycat_test)) * 100

print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)
#36.75463751878325
#33.75402423022601
#36.615710155370266

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()
ax.scatter(predscat,ycat_test)
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()

def calculate_bias(pred, truev):
    # Вычисляем смещение (BIAS) данных относительно среднего значения
    bias = sum(pred - truev) / len(truev)
    return bias

# Пример использования функции
bias = calculate_bias(predscat, ycat_test)
print("BIAS данных:", bias)


# XG BOOST для НЕнормализированных данных

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
import plotly.express as px

Xgb = df2.values[:,(0,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]
Ygb = df2.values[:, 1]


Xgb_train, Xgb_test, ygb_train, ygb_test = train_test_split(Xgb, Ygb, test_size=0.1, random_state=0)

bst = XGBRegressor(n_estimators=100, max_depth=16, objective='reg:squarederror',eval_metric="rmse")

# fit model
bst.fit(Xgb_train, ygb_train)

predsgb = bst.predict(Xgb_test)

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(ygb_test, predsgb, squared=False)
mae = mean_absolute_error(ygb_test, predsgb)
# Рассчет MAPE
mape = np.mean(np.abs((ygb_test - predsgb) / ygb_test)) * 100

print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)
#45.1036560331101
#39.75442084840644
#39.989281025981214

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()
ax.scatter(predsgb, ygb_test)
plt.show()


# Данные с нормализацией

Данные с нормализацией

dfn = df2.copy()
dfn["Total Sales"] = 1000 * ((dfn["Total Sales"] - dfn["Total Sales"].min()) / (dfn["Total Sales"].max() - dfn["Total Sales"].min()))
dfn

Xn= dfn.values[:,(0,2,3,4,5,6,7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)]
Yn= dfn.values[:, 1]


Xn_train, Xn_test, yn_train, yn_test = train_test_split(Xn, Yn, test_size=0.15, random_state=1)

# CATBOOST для нормализированных данных

from catboost import Pool, CatBoostRegressor

# initialize Pool
train_pool = Pool(Xn_train, yn_train)
test_pool = Pool(Xn_test, yn_test)

# specify the training parameters
model = CatBoostRegressor(iterations=2500, loss_function='MAE', verbose=100)
#train the model
model.fit(train_pool)
# make the prediction using the resulting model
predsn = model.predict(test_pool)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

x, y = np.random.random((2, 100))*2
fig, ax = plt.subplots()
ax.scatter(predsn, yn_test)
line = mlines.Line2D([0, 1], [0, 1], color='red')
transform = ax.transAxes
line.set_transform(transform)
ax.add_line(line)
plt.show()

from sklearn.metrics import mean_squared_error, mean_absolute_error

rmse = mean_squared_error(yn_test, predsn, squared=False)
mae = mean_absolute_error(yn_test, predsn)
# Рассчет MAPE
mape = np.mean((np.abs(yn_test - predsn)) / (yn_test)) * 100

print("MAE:", mae)
print("MAPE:", mape)
print("RMSE:", rmse)
#35.109849965726916
#36.17786197587769
#35.84043296502193

bias = calculate_bias(predsn, yn_test)
print("BIAS данных:", bias)

# Линейная Регрессия Для нормализированных данных

lr = LinearRegression()
lr.fit(Xn_train,yn_train)
y_pred = lr.predict(Xn_test)


# Рассчет MSE
mse = np.mean((yn_test - y_pred)**2)

# Рассчет MAPE
mape = np.mean(np.abs((yn_test - y_pred)) / (yn_test)) * 100

print("MSE:", mse)
print("MAPE:", mape)
#87.54144278462634
#98.21869232813982
#89.93882815686327

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

x, y = np.random.random((2, 100))*2
fig, ax = plt.subplots()
ax.scatter(y_pred, yn_test)

plt.show()

# XGBOOST для нормализированных данных
```
