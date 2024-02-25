# AirQualityRegression
[![Open Issues](https://img.shields.io/github/issues-raw/jgullinkala/AirQualityRegression)](https://github.com/jgullinkala/AirQualityRegression/issues)

## Quick Install
Install the application by running requirements.txt:
```
pip install -r requirements.txt
```

## Project Description
AirQuality Italy UCI dataset is a multivariate time-series dataset that reports the hourly concentrations of air pollutants in an Italian city. The dataset contains 9358 instances and 15 attributes. The dataset is available at the UCI Machine Learning Repository

The dataset contains the following attributes:

1. Date: Date (DD/MM/YYYY)
2. Time: Time (HH.MM.SS)
3. CO(GT): True hourly averaged concentration CO in mg/m^3 (reference analyzer)
4. PT08.S1(CO): (tin oxide) hourly averaged sensor response (nominally CO targeted)
5. NMHC(GT): True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
6. C6H6(GT): True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
7. PT08.S2(NMHC): (titania) hourly averaged sensor response (nominally NMHC targeted)
8. NOx(GT): True hourly averaged NOx concentration in ppb (reference analyzer)
9. PT08.S3(NOx): (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
10. NO2(GT): True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
11. PT08.S4(NO2): (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
12. PT08.S5(O3): (indium oxide) hourly averaged sensor response (nominally O3 targeted)
13. T: Temperature in °C
14. RH: Relative Humidity (%)
15. AH: Absolute Humidity

The dataset contains missing values and the target variable is Date. It is choosen as target variable because it is a time-seires dataset and we are interested in predicting the concentration of other pollutants based on the date.

## The code covers the following steps:

1. Reading the dataset
```python
import pandas as pd
air_quality_df = pd.read_csv('https://archive.ics.uci.edu/static/public/360/data.csv')
```
2. Cleaning the data
```python
df = df.replace(-200, np.nan)
df = df.fillna(df.mean())
```
3. Exploratory Data Analysis
```python
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='viridis')
plt.show()
```
4. Feature Selection (PCA)
```python
from sklearn.decomposition import PCA  
from sklearn.preprocessing import StandardScaler
X = data_features.drop(columns=['Date', 'Time','Peak_hours'])
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(X)
pca = PCA()  
X_pca = pca.fit(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio.cumsum()
```
5. Model Building
```python
from sklearn.linear_model import LinearRegression
X = data_features.drop('Date', axis=1)
y = data_features['Date']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
```
6. Model Evaluation
```python
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test_pca)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print('RMSE of Linear Regression with PCA:', rmse)
r2 = r2_score(y_test, y_pred)
print('R-squared of Linear Regression with PCA:', r2)
```

## The code is written in Python and uses the following libraries:

1. pandas
2. numpy
3. matplotlib
4. seaborn
5. sklearn

## The dataset is available at the UCI Machine Learning Repository. The dataset is available at the following URL:
https://archive.ics.uci.edu/ml/datasets/Air+Quality

## The code is available at the following URL:
https://github.com/jgullinkala/AirQualityRegression

