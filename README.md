
# Project Overview
- Predict Rotten Tomatoes audience ratings using machine learning models.
- Dataset includes features like movie title, rating, tomatometer rating, runtime, and critics’ consensus.
- Target variable: `audience_rating`.

# Key Features
- **Exploratory Data Analysis (EDA)**: Visualize distributions, correlations, and genre/studio trends.
- **Feature Engineering**:
  - Created `date_difference` (difference between release and streaming dates).
  - Performed sentiment analysis on `critics_consensus` to generate a `review_sentiment` feature.
- **Data Preprocessing**:
  - Encoded categorical features (`rating`, `tomatometer_status`).
  - Scaled continuous features using `StandardScaler`.
  - Handled missing values in `runtime_in_minutes`.
- **Model Training and Evaluation**:
  - Models: Random Forest, XGBoost, LightGBM, SVR, and more.
  - Metrics: Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R².
  - Results stored and visualized for comparison.
- **Streamlit App**:
  - Interactive interface to input movie details.
  - Supports model selection and real-time predictions.
  - Displays evaluation metrics and results.

# Deployment
- Models saved using `joblib` for deployment.
- Streamlit app processes inputs, encodes features, performs sentiment analysis, and scales data for predictions.
