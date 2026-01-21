import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.datasets import fetch_california_housing
import numpy as np

# Load data
data = fetch_california_housing(as_frame=True)
df = data.frame

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Column separation
cat_cols = X.select_dtypes(include=['object', 'category']).columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessing = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), num_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Linear Regression Pipeline
model = Pipeline([
    ('preprocessing', preprocessing),
    ('regressor', GradientBoostingRegressor(loss="huber",learning_rate=0.1,n_estimators=5))
])

lower = Pipeline([
    ('preprocesing',preprocessing),
    ('regressor',GradientBoostingRegressor(loss='quantile', n_estimators=5,alpha=0.1))
])

upper = Pipeline([
    ('preprocesing',preprocessing),
    ('regressor',GradientBoostingRegressor(loss='quantile', n_estimators=5,alpha=0.9))
])
# Train
model.fit(X_train, y_train)
lower.fit(X_train, y_train)
upper.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
lower_pred = lower.predict(X_test)
upper_pred = upper.predict(X_test)
# Evaluation
print("R²:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", mean_squared_error(y_test, y_pred))

df = pd.DataFrame({
    "prediction": y_pred,
    "lower_bound": lower_pred,
    "upper_bound": upper_pred
})

print(df.head())

df["interval_width"] = df["upper_bound"] - df["lower_bound"]
print(df[["interval_width"]].describe())

print(y_test.describe())