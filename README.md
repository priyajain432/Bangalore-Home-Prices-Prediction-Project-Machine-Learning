# 🏠 Bangalore Home Prices Prediction Project

Predicting home prices in Bangalore using regression models in Python. This project focuses on **data cleaning, feature engineering**, and building an **accurate predictive model** for real estate pricing.

---

## 🎯 Objective

The goal of this project is to create a **machine learning model** that can predict the **price of residential properties in Bangalore** based on features like location, square footage, number of bedrooms, and bathrooms. 

This project enables:

- **Accurate price prediction 💰** for buyers, sellers, and real estate agents.
- **Insights into property market trends 📊**.
- **Data-driven decision making 🧠** for real estate investments.

---

## 📦 Dataset

- **Source:** Kaggle / Local CSV  
- **File Name:** `bengaluru_house_prices.csv`  
- **Description:** The dataset contains real estate property listings in Bangalore with the following columns:

| Column | Description |
|--------|-------------|
| area_type | Type of area (Super built-up, Plot, etc.) |
| location | Location of the property |
| size | Number of bedrooms (e.g., 2 BHK) |
| total_sqft | Total square footage of the property |
| bath | Number of bathrooms |
| balcony | Number of balconies |
| price | Price of the property in Lakhs |

---

## 🧹 Data Cleaning & Feature Engineering

- Removed irrelevant features: `area_type`, `society`, `balcony`, `availability`.
- Dropped rows with missing values (`NaN`).
- Converted `size` to numerical **BHK** feature.
- Converted `total_sqft` to float; averaged ranges like `2100-2850`.
- Added **Price per Square Feet** feature:

- Reduced number of **locations** by tagging low-frequency locations as `"other"`.
- Removed outliers based on:
- Minimum 300 sqft per BHK.
- Price per sqft deviations using mean ± std.
- Logical inconsistencies in bathrooms vs bedrooms.

---

## 📊 Exploratory Data Analysis (EDA)

- Scatter plots for **2 BHK vs 3 BHK properties** in popular locations (e.g., Rajaji Nagar, Hebbal).  
- Histograms to analyze distribution of `price_per_sqft` and `bathrooms`.  
- Visualized outlier removal for better model accuracy.

---

## 🔢 Encoding Categorical Features

- Applied **One Hot Encoding** to the `location` column.  
- Dropped `"other"` column to avoid multicollinearity.  

---

## 🏗️ Model Building

- **Features:** `total_sqft`, `bath`, `bhk`, `location dummies`  
- **Target:** `price`  
- **Train/Test Split:** 80% train, 20% test  
- **Algorithms Tested:** 
- Linear Regression  
- Lasso Regression  
- Decision Tree Regressor  

- **Cross-validation:** ShuffleSplit (5 splits)  
- **Best Model:** Linear Regression (accuracy > 80%)

---

## 🔍 Model Evaluation

| Model | Best Score | Notes |
|-------|------------|-------|
| Linear Regression | ~0.85 | Best performance, chosen for deployment |
| Lasso | ~0.82 | Slightly lower, regularized model |
| Decision Tree | ~0.80 | Slight overfitting |

---

## 🛠️ Predicting Prices

Example usage of the trained Linear Regression model:

```python
def predict_price(location, sqft, bath, bhk):
  # Predict price based on input features
  pass

predict_price('1st Phase JP Nagar', 1000, 2, 2)
predict_price('Indira Nagar', 1000, 3, 3)
