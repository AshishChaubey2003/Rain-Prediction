# Rain-Prediction
# 🌦️ Weather Prediction Project (RainTomorrow)

This project focuses on predicting whether it will rain the next day (`RainTomorrow`) using the Australian weather dataset. Various data preprocessing steps and machine learning models are applied to ensure accurate and reliable predictions.

## 📁 Dataset
- **Source**: [Kaggle - Australian Weather Dataset](https://www.kaggle.com/datasets/jsphyg/weather-dataset-rattle-package)
- **Size**: ~145,000 rows
- **Target Variable**: `RainTomorrow` (Yes/No)

## 🧰 Tools & Libraries Used
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost, CatBoost, Random Forest, SVM
- SMOTE (for class imbalance handling)

## 🔧 Key Features
- Handled missing values with techniques like random sampling and mode/median imputation.
- Encoded categorical variables using mapping and one-hot encoding.
- Removed outliers using IQR method to enhance model robustness.
- Performed feature engineering (e.g., date extraction, mapping wind directions).
- Correlation analysis and data visualization using heatmaps and boxplots.

## 🤖 Models Trained
- CatBoost Classifier
- Random Forest Classifier
- Logistic Regression
- XGBoost Classifier
- K-Nearest Neighbors
- Naive Bayes
- Support Vector Machine (SVM)

## 📊 Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- ROC Curve

## 📌 Results
CatBoost and RandomForest performed best in terms of AUC and overall accuracy. SMOTE significantly improved performance on imbalanced data.

## 📁 Output
Preprocessed dataset saved as `preprocessed_1.csv`. Trained models can be saved using `joblib` for deployment.

## 🚀 How to Run
1. Clone the repo and place the `weatherAUS.csv` in the project folder.
2. Run the Python script in your IDE or Jupyter Notebook.
3. Evaluate model performance and visualize results.

---

### 📫 Author
**Ashish Kumar Chaubey**  
Final Year B.Tech CSE Student  
[LinkedIn](#) | [GitHub](#)

