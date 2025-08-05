import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from math import radians, sin, cos, asin, sqrt
import joblib

df = pd.read_csv('train.csv')

print("Initial Shape:", df.shape)
print(df.info())

print(df.isnull().sum())
print(f"Duplicates: {df.duplicated().sum()}")


df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
df['dropoff_datetime'] = pd.to_datetime(df['dropoff_datetime'])

df = df[(df['trip_duration'] >= 10) & (df['trip_duration'] <= 21600)]

df['hour'] = df['pickup_datetime'].dt.hour
df['day_of_week'] = df['pickup_datetime'].dt.dayofweek
df['month'] = df['pickup_datetime'].dt.month
df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
df['is_rush_hour'] = ((df['hour'].between(7,10)) | (df['hour'].between(16,19))).astype(int)

df['log_trip_duration'] = np.log1p(df['trip_duration'])

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

df['distance_km'] = haversine_vectorized(
    df['pickup_longitude'], df['pickup_latitude'],
    df['dropoff_longitude'], df['dropoff_latitude']
)

df['manhattan_distance'] = (abs(df['dropoff_longitude'] - df['pickup_longitude']) +
                             abs(df['dropoff_latitude'] - df['pickup_latitude']))

df['speed_kmh'] = (df['distance_km'] / (df['trip_duration'] / 3600)).replace([np.inf, -np.inf], 0)

df = df[df['distance_km'] < 100]

features = ['passenger_count', 'vendor_id', 'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
            'distance_km', 'manhattan_distance', 'speed_kmh']
X = df[features]
y = df['log_trip_duration']

categorical_cols = ['vendor_id']
numeric_cols = [col for col in features if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model --> XGBoost Regressor
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', model)
])

pipeline.fit(X_train, y_train)

y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_original = np.expm1(y_test)

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
r2 = r2_score(y_test_original, y_pred)

print(f"Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

joblib.dump(pipeline, 'advancedModel_pipeline.pkl')
print("Pipeline saved as 'advancedModel_pipeline.pkl'")

#  ( Feature Importance )
# model.fit(X_train, y_train)  # Fit raw model for importance
# xgb.plot_importance(model)
# plt.show()
