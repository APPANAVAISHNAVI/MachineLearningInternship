import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

train_data_path = r'C:\Users\vaish\PycharmProjects\MachineLearning\train.csv'
train_data = pd.read_csv(train_data_path)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
X = train_data[features].copy()
y = train_data['SalePrice']
X.loc[:, 'TotalBath'] = X['FullBath'] + 0.5 * X['HalfBath']
X = X.drop(['FullBath', 'HalfBath'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)