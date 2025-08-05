import pandas as pd
import numpy as np
import joblib
from math import radians, sin, cos, asin, sqrt

# -------------------------------------------------------------------------------------------------------
# 1- Kaggle Test Data Preparation ----> ( BaseLine_Model.py )
# -------------------------------------------------------------------------------------------------------

test_df = pd.read_csv('test.csv')

test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

test_df['hour'] = test_df['pickup_datetime'].dt.hour
test_df['day_of_week'] = test_df['pickup_datetime'].dt.dayofweek

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r

test_df['distance_km'] = haversine_vectorized(
    test_df['pickup_longitude'], test_df['pickup_latitude'],
    test_df['dropoff_longitude'], test_df['dropoff_latitude']
)

pipeline = joblib.load('taxi_duration_pipeline.pkl')

features = ['passenger_count', 'vendor_id', 'hour', 'day_of_week', 'distance_km']
X_test_final = test_df[features]

y_pred_log = pipeline.predict(X_test_final)
y_pred = np.expm1(y_pred_log)  # Convert back from log scale

# Save predictions to submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'trip_duration': y_pred
})

submission.to_csv('submission_BaseLine_Model.csv', index=False)
print("✅ Predictions saved to submission_BaseLine_Model.csv")


# -------------------------------------------------------------------------------------------------------
# 2- Kaggle Test Data Preparation ----> advancedModel.py
# -------------------------------------------------------------------------------------------------------

# test_df = pd.read_csv('test.csv')

# test_df['pickup_datetime'] = pd.to_datetime(test_df['pickup_datetime'])

# test_df['hour'] = test_df['pickup_datetime'].dt.hour
# test_df['day_of_week'] = test_df['pickup_datetime'].dt.dayofweek
# test_df['month'] = test_df['pickup_datetime'].dt.month
# test_df['is_weekend'] = test_df['day_of_week'].isin([5, 6]).astype(int)
# test_df['is_rush_hour'] = ((test_df['hour'].between(7,10)) | (test_df['hour'].between(16,19))).astype(int)

# def haversine_vectorized(lon1, lat1, lon2, lat2):
#     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
#     dlon = lon2 - lon1
#     dlat = lat2 - lat1
#     a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
#     c = 2 * np.arcsin(np.sqrt(a))
#     r = 6371
#     return c * r

# test_df['distance_km'] = haversine_vectorized(
#     test_df['pickup_longitude'], test_df['pickup_latitude'],
#     test_df['dropoff_longitude'], test_df['dropoff_latitude']
# )

# test_df['manhattan_distance'] = (abs(test_df['dropoff_longitude'] - test_df['pickup_longitude']) +
#                                  abs(test_df['dropoff_latitude'] - test_df['pickup_latitude']))


# test_df['speed_kmh'] = 0 

# features = ['passenger_count', 'vendor_id', 'hour', 'day_of_week', 'month',
#             'is_weekend', 'is_rush_hour', 'distance_km', 'manhattan_distance', 'speed_kmh']
# X_test_kaggle = test_df[features]

# pipeline = joblib.load('advancedModel_pipeline.pkl')

# y_pred_log = pipeline.predict(X_test_kaggle)
# y_pred = np.expm1(y_pred_log)

# submission = pd.DataFrame({
#     'id': test_df['id'],
#     'trip_duration': y_pred
# })

# submission.to_csv('submission_advancedModel.csv', index=False)
# print("Submission file saved as 'submission.csv'")
