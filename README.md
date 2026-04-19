# 🏦 Loan Prediction 2.0

A machine learning application that predicts loan eligibility with explainable AI (SHAP) insights. This project combines predictive modeling with interpretability to help understand why loans are approved or rejected.

## 🎯 Features

- **Machine Learning Predictions**: Predicts loan eligibility based on customer financial profiles
- **SHAP Explainability**: Provides detailed analysis of feature contributions to each prediction
- **Interactive Dashboard**: Streamlit-based web interface for real-time predictions
- **Robust Visualizations**: Multiple SHAP visualization options (waterfall, bar charts, feature importance)
- **Data Processing Pipeline**: Automated feature engineering and data preprocessing
- **Model Persistence**: Saves trained models and SHAP explainers for reproducibility

## 🛠 Tech Stack

- **Python 3.8+**
- **scikit-learn**: Machine Learning
- **SHAP v0.20+**: Model Explainability
- **Streamlit**: Interactive Dashboard
- **Pandas & NumPy**: Data Processing
- **Matplotlib**: Visualization

## 📁 Project Structure

```
├── app.py                      # Streamlit dashboard application
├── train_model.py              # Model training and evaluation script
├── data_processing.py          # Data cleaning and feature engineering
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── data/
│   ├── train.csv              # Raw training dataset
│   └── processed_train.csv    # Processed dataset
└── models/
    ├── loan_model.pkl         # Trained ML model
    ├── explainer.pkl          # SHAP explainer object
    └── feature_names.pkl      # Feature names for consistency
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/Loan-Prediction-2.0.git
   cd Loan-Prediction-2.0
   ```

2. **Create virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. **Train the model** (first time or after data updates):
   ```bash
   python train_model.py
   ```
   This will:
   - Load and process training data
   - Train the model
   - Generate SHAP explainer
   - Save artifacts to `models/` directory

2. **Launch the dashboard**:
   ```bash
   streamlit run app.py
   ```
   - Open your browser to `http://localhost:8501`
   - Enter customer details in the sidebar
   - Click "Predict Eligibility" to see predictions and explanations

## 📊 Understanding the Dashboard

- **Prediction Result**: Shows whether the loan is approved or rejected
- **Confidence Score**: Displays model confidence percentage
- **SHAP Explanation**: 
  - **Waterfall Plot**: Shows cumulative feature contributions (base value → prediction)
  - **Feature Importance Bar Chart**: Top 10 most impactful features
  - **Feature Values Table**: Raw SHAP values for all features

## 🔄 Model Pipeline

1. **Data Processing** → Feature engineering, encoding, scaling
2. **Model Training** → Classification model (defaults to Logistic Regression/Random Forest)
3. **SHAP Analysis** → Generate SHAP explainer for interpretability
4. **Dashboard Display** → Real-time predictions with explanations

## 📝 Version History

### Phase 1 — Data Pipeline (Current)
- ✅ SHAP v0.20 compatibility update
- ✅ Robust visualization with intelligent fallbacks
- ✅ Waterfall plot primary visualization
- ✅ Feature importance bar charts as fallback
- ✅ Git repository and .gitignore setup

## 🐛 Troubleshooting

**SHAP Visualization Issues**:
- If force plot fails, the app automatically falls back to waterfall plot
- If waterfall plot fails, feature importance bar chart is shown
- If all fail, raw SHAP values are displayed in a table

**Model Not Found**:
- Run `python train_model.py` first to generate model artifacts

**Dependencies Issues**:
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt` again

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

Created as a personal project for loan eligibility prediction and machine learning explainability.

## 🙋 Support

For issues or questions, please open an issue on GitHub or contact the project maintainer.
