# Heart Disease Analytics Dashboard

A production-ready healthcare analytics application featuring machine learning predictions, explainable AI, and comprehensive data visualizations for heart disease risk assessment.

## Project Overview

This project implements a complete ML pipeline for heart disease prediction using the UCI Heart Disease dataset. It includes data preprocessing, model training with scikit-learn, and an interactive web dashboard built with Streamlit that provides:

- **Batch Predictions**: Upload patient data and get instant predictions for multiple patients
- **Explainable AI**: SHAP (SHapley Additive exPlanations) values for model interpretability
- **Interactive Visualizations**: EDA charts, group comparisons, and patient-level analysis
- **Robust Data Handling**: Automatic feature imputation and multiple file format support

## Key Features

### Machine Learning Pipeline
- **Random Forest Classifier** with 150 estimators achieving ~85% accuracy
- **Preprocessing Pipeline**: Automated feature scaling, imputation, and one-hot encoding
- **Cross-validated Training**: Stratified train-test split with reproducible random state
- **Feature Engineering**: Comprehensive handling of numeric and categorical features

### Dashboard Capabilities
- **Exploratory Data Analysis**: Histograms, scatter plots, correlation heatmaps, box plots, violin plots
- **Batch Prediction**: Process CSV files with automatic missing value handling
- **SHAP Explanations**: Individual patient feature contribution analysis with waterfall plots
- **Group Analytics**: Compare predicted positive vs negative cohorts with statistical insights
- **Export Functionality**: Download predictions and analysis results

### Code Quality
- Unit tests with pytest
- CI/CD pipeline with GitHub Actions
- Modular architecture with clear separation of concerns
- Type hints and comprehensive error handling

## Technology Stack

**Core:**
- Python 3.10+
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, RandomForestClassifier)

**Visualization:**
- Streamlit
- Plotly Express
- Seaborn, Matplotlib

**Explainability:**
- SHAP for model interpretability

**Testing & DevOps:**
- pytest
- GitHub Actions
- Docker support

## Project Structure

```
heart_disease_dashboard/
├── src/
│   ├── app/
│   │   └── streamlit_app.py          # Main dashboard application
│   ├── models/
│   │   ├── train_model.py            # Model training pipeline
│   │   └── predict.py                # Prediction utilities
│   ├── data/
│   │   └── preprocess.py             # Data cleaning functions
│   └── visualization/
│       └── eda_plots.py              # Chart generation utilities
├── data/
│   ├── raw/                          # Original dataset
│   ├── processed/                    # Cleaned dataset
│   └── sample_input.csv              # Example input file
├── models/
│   ├── rf_pipeline.joblib            # Trained model artifact
│   ├── metrics.json                  # Model performance metrics
│   └── feature_importances.csv       # Feature importance scores
├── tests/                            # Unit tests
├── scripts/                          # Helper scripts
└── .github/workflows/                # CI/CD configuration
```

## Getting Started

### Prerequisites
- Python 3.10 or higher
- pip package manager
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Siddhant-21-03/Heart-Disease.git
cd Heart-Disease
```

2. **Set up virtual environment**
```bash
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Training the Model

Train a new model with custom parameters:

```bash
# Windows
python src\models\train_model.py --input data\raw\heart_disease_uci.csv --output models --n-estimators 150

# Linux/Mac
python src/models/train_model.py --input data/raw/heart_disease_uci.csv --output models --n-estimators 150
```

Or use the provided training scripts:
```bash
# Windows
scripts\train.bat

# Linux/Mac
bash scripts/train.sh
```

### Running the Dashboard

```bash
streamlit run src/app/streamlit_app.py
```

The dashboard will open at `http://localhost:8501`

### Running Tests

```bash
pytest
```

## Usage Guide

### 1. Upload Data
- Navigate to the **Upload** tab
- Upload a CSV file with patient data
- View automatic batch predictions and group insights

### 2. Explore Data
- Use the **EDA** tab for data visualization
- Compare predicted positive vs negative groups
- Generate histograms, box plots, and correlation heatmaps

### 3. Model Insights
- View **SHAP explanations** for individual patients
- Understand feature contributions with waterfall plots
- Analyze which features drive predictions

### 4. Single Prediction
- Enter patient information in the **Predict** tab
- Get instant risk assessment
- View personalized insights

## Model Performance

- **Accuracy**: ~85%
- **Algorithm**: Random Forest (150 trees)
- **Validation**: Stratified 80/20 train-test split
- **Features**: 13 clinical features including age, sex, chest pain type, cholesterol, etc.

## Dataset

The project uses the UCI Heart Disease dataset containing 303 patient records with 14 attributes. The target variable indicates the presence of heart disease (binary classification).

**Key Features:**
- Age, Sex, Chest Pain Type
- Resting Blood Pressure, Cholesterol
- Fasting Blood Sugar, ECG Results
- Maximum Heart Rate, Exercise Angina
- ST Depression, Slope, Vessels, Thalassemia

## Contributing

This is a portfolio project, but suggestions and feedback are welcome through GitHub issues.

## License

This project is for educational and portfolio purposes.

## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- SHAP library for explainability features
- Streamlit for the interactive dashboard framework
