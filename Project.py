import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.metrics import mean_squared_error,r2_score
df = pd.read_csv("Used_Bikes.csv")
df.dropna(inplace=True)
numeric_cols = ['kms_driven', 'age', 'power']
numerics_df = df[numeric_cols].reset_index(drop=True)
encoder_owner = OneHotEncoder(sparse_output = False,drop = None)
encoder_city = OneHotEncoder(sparse_output=False, drop=None)
encoder_brand = OneHotEncoder(sparse_output=False, drop=None)
owner_en = encoder_owner.fit_transform(df[['owner']])
city = encoder_city.fit_transform(df[['city']])
owner_df = pd.DataFrame(owner_en,columns=encoder_owner.get_feature_names_out(['owner']))
city_df = pd.DataFrame(city,columns=encoder_city.get_feature_names_out(['city']))
df_encoded = pd.concat([numerics_df, owner_df, city_df], axis=1)
# df_encoded['price'] = df['price'].values
x = df_encoded
y = df['price'].values
print(df_encoded.head())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7)
start = time.time()
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
## Random Forest Regressor ##
start = time.time()
model = RandomForestRegressor(n_jobs=-1)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid,cv=3, scoring='r2', verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)
end = time.time()
y_pred = grid_search.predict(x_test)
y_train_pred = grid_search.predict(x_train)
r2_train = r2_score(y_train,y_train_pred)
r2 = r2_score(y_test, y_pred)
print(grid_search.best_params_)
print("Training time: {:.2f} seconds".format(end - start))
print("R² Train Score:", r2_train*100)
print("R² Score:", r2*100)
## Logistic Regression ##
df['price_category'] = pd.qcut(df['price'],q=4,labels=[0,1,2,3])
y_cl = df['price_category'].astype(int)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train_cls, x_test_cls, y_train_cls, y_test_cls = train_test_split(x_scaled, y_cl, test_size=0.7)
start1 = time.time()
log = LogisticRegression(max_iter=100000)
log.fit(x_train_cls, y_train_cls)
end1 = time.time()
y_train_pred_log = log.predict(x_train_cls)
y_test_pred_log = log.predict(x_test_cls)
train_accuracy = accuracy_score(y_train_cls, y_train_pred_log)
test_accuracy = accuracy_score(y_test_cls, y_test_pred_log)
print("Training time: {:.2f} seconds".format(end1 - start1))
print("Accuracy Train Score:", train_accuracy * 100)
print("Accuracy Test Score:", test_accuracy * 100)

trained_model = grid_search
encoder_owner_fitted = encoder_owner
encoder_city_fitted = encoder_city
scaler_fitted = scaler
logistic_model = log  