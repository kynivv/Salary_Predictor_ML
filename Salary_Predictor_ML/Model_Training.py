import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score as evs

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor

# Data Import
df = pd.read_csv('Salary_Data_Based_country_and_race.csv')


# EDA
print(df.info())
print(df.isnull().sum())


# Data Preproccesing
df.dropna(inplace= True)

le = LabelEncoder()

for c in df.columns:
    if df[c].dtype == 'object':
        df[c] = le.fit_transform(df[c])

df = df.drop('Unnamed: 0', axis=1)


# Train Test Split
target = df['Salary']
features = df.drop('Salary', axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.20)


# Model Training
models = [Ridge(), LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor()]

for m in models:
    print(m)

    m.fit(X_train, Y_train)

    pred_train = m.predict(X_train)
    print(f'Train Accuracy: {evs(Y_train, pred_train)}')

    pred_test = m.predict(X_test)
    print(f'Test Accuracy: {evs(Y_test, pred_test)}')
    print('#' * 100)

