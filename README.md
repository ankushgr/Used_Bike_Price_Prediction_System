🏍️ Used Bike Price Prediction

This project is modelled to predict the prices of used bikes and classify them into different price ranges, given features like kilometers driven, age, engine power, ownership history, location, etc., using machine learning.

📌 Features

Prediction of bike price using Random Forest Regression

Logistic Regression for bike pricing categories.

One-Hot Encoding for Categorical data support Handler - For handling the Categorical Data.

Utilizes Standard Scaling for classification

Does GridSearchCV for tuning hyperparameter

Assessment of the model using R² Score, Accuracy and Classification Report

🧾 Dataset

The dataset Used_Bikes. csv includes fields like:

kms_driven

age

power

owner

city

brand

price

There are some rows with an NA value and are not discarded during the preprocessing.

🧠 Models Used

🔸 Random Forest Regressor.

It is used as to predict the continuous value of the price of the bike.

Hyperparameters were tuned by GridSearchCV:

n_estimators

max_depth

min_samples_split

min_samples_leaf

🔸 Logistic Regression

Previously, when classifying bike prices into 4 tiers based on the price using quantile based binning - (qcut).

🛠️ How to Run

Clone the repository:

git clone https://github.com/ankushgr/Used-Bike-Price-Prediction.git

cd Used-Bike-Price-Prediction
