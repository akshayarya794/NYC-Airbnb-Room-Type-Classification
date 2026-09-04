# NYC Airbnb Room Type Classification

An end-to-end machine learning project predicting the listing room type (`Entire home/apt`, `Private room`, or `Shared room`) using tabular listing attributes such as pricing, geographic coordinates, availability, and review dynamics from the [New York City Airbnb Open Data](https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data).

---

## Project Overview

The goal of this project is to automate and standardize listing categorization on short-term rental platforms. By framing the task as a multi-class tabular classification problem, the notebook covers the complete ML lifecycle: from exploratory data analysis and outlier capping to leakage-free preprocessing, multi-model evaluation, hyperparameter tuning, and model serialization.

### Highlights
- **Class Imbalance Handling:** Prioritized Macro-F1 score and balanced class weights to account for minority categories (`Shared room`).
- **Data Hygiene:** 99th percentile capping on heavy-tailed features (`price`, `minimum_nights`) to prevent outlier-driven distortion.
- **Leakage-Free Architecture:** Bundled imputation, scaling, and one-hot encoding into a unified `ColumnTransformer` evaluated via stratified cross-validation.
- **Production-Ready Artifact:** Exported the end-to-end inference pipeline as a single compressed `.pkl` object.

---

## Dataset Description

The analysis uses the New York City Airbnb 2019 dataset containing 48,895 listings and 16 original features.

- **Target Variable:** `room_type` (`Entire home/apt`, `Private room`, `Shared room`)
- **Dropped High-Cardinality/ID Fields:** `id`, `name`, `host_id`, `host_name`, `last_review`
- **Features Used:**
  - *Numerical:* `latitude`, `longitude`, `price`, `minimum_nights`, `number_of_reviews`, `reviews_per_month`, `calculated_host_listings_count`, `availability_365`
  - *Categorical:* `neighbourhood_group`, `neighbourhood`

 Raw Input Data
│
├── Numeric Features ────► Median Imputer ────► StandardScaler
│
└── Categorical Features ─► Most Frequent ────► OneHotEncoder (handle_unknown='ignore')
│
▼
RandomForestClassifier
(class_weight='balanced')

## Model Benchmarking & Results

Models were evaluated using 3-fold stratified cross-validation on the training set:

| Model | CV Accuracy | CV Macro-F1 |
| :--- | :--- | :--- |
| **Logistic Regression** | 0.659 | 0.522 |
| **Decision Tree** | 0.782 | 0.647 |
| **Gradient Boosting** | 0.850 | 0.705 |
| **Random Forest (Baseline)** | 0.851 | 0.715 |
| **Random Forest (Tuned)** | **0.855** | **0.738** |

### Optimal Hyperparameters
- `n_estimators`: 200
- `min_samples_split`: 10
- `max_depth`: `None`
- `class_weight`: `balanced`

### Test Set Performance (Unseen Data)
- **Accuracy:** 85.54%
- **Macro-F1:** 0.7380

---

## Project Structure

```text
├── AB_NYC_2019.csv                               # Source dataset
├── nyc_airbnb_room_type_classification.ipynb     # Jupyter/Colab notebook
├── Model_Pipeline.pkl                            # Serialized end-to-end pipeline
└── README.md                                     # Documentation

Quickstart & Inference
1. Installation
Bash
git clone [https://github.com/](https://github.com/)<your-username>/nyc-airbnb-classification.git
cd nyc-airbnb-classification
pip install numpy pandas scikit-learn joblib
2. Loading the Model for Inference
Python
import joblib
import pandas as pd

# Load the saved pipeline artifact
model = joblib.load("Model_Pipeline.pkl")

# Prepare new listing sample (raw data format)
sample_listing = pd.DataFrame([{
    "neighbourhood_group": "Manhattan",
    "neighbourhood": "Hell's Kitchen",
    "latitude": 40.75505,
    "longitude": -73.99531,
    "price": 120,
    "minimum_nights": 7,
    "number_of_reviews": 6,
    "reviews_per_month": 0.50,
    "calculated_host_listings_count": 47,
    "availability_365": 332
}])

# Inference without manual preprocessing
prediction = model.predict(sample_listing)
print(f"Predicted Room Type: {prediction[0]}")
Future Improvements
Incorporate NLP features (TF-IDF or embeddings) from listing name.

Experiment with cost-sensitive XGBoost or LightGBM variants.

Deploy the pipeline artifact via a lightweight FastAPI or Flask prediction microservice.


---

## Pipeline Architecture
