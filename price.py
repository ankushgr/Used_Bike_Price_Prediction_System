import pandas as pd
from Project import trained_model, encoder_owner_fitted, encoder_city_fitted, scaler_fitted, logistic_model
# New bike data
new_bike = {
    'kms_driven': 5000,
    'age': 1,
    'power': 350,
    'owner': 'First Owner',
    'city': 'Zirakpur'
}

# Convert to DataFrame
new_df = pd.DataFrame([new_bike])

# Encode categorical variables using the fitted encoders
owner_encoded = encoder_owner_fitted.transform(new_df[['owner']])
city_encoded = encoder_city_fitted.transform(new_df[['city']])
owner_df_new = pd.DataFrame(owner_encoded, columns=encoder_owner_fitted.get_feature_names_out(['owner']))
city_df_new = pd.DataFrame(city_encoded, columns=encoder_city_fitted.get_feature_names_out(['city']))

# Combine with numeric columns
final_input = pd.concat([
    new_df[['kms_driven', 'age', 'power']].reset_index(drop=True),
    owner_df_new,
    city_df_new
], axis=1)

# Predict price
predicted_price = trained_model.predict(final_input)
print(f"\nPredicted Price: ₹{predicted_price[0]:,.2f}")

final_input_scaled = scaler_fitted.transform(final_input)
predicted_category = logistic_model.predict(final_input_scaled)[0]

# Map category to descriptive label (optional)
category_mapping = {
    0: "Low Price Range",
    1: "Mid-Low Price Range",
    2: "Mid-High Price Range",
    3: "High Price Range"
}
category_label = category_mapping.get(predicted_category, "Unknown")

print(f"Predicted Price Category (Logistic): {category_label} (Category {predicted_category})")
