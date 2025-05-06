# 🏦 Lending Club Loan Grade Prediction

## 📊 Overview

This project analyzes Lending Club data to build prediction models for loan acceptance, loan grade, and loan sub-grade based on customers' credit and history records. The models can assist financial institutions in assessing loan applications and determining appropriate loan grades for approved applications.

## 📚 Dataset

The analysis uses Lending Club data which includes:

- Customer credit history
- Loan application details
- Amount requested
- Loan titles
- Various financial parameters
- Employment information
- Geographic data

## 📗 Notebooks

- [1_data_cleaning.ipynb](https://github.com/MeiChieh/lending-club-loan-grade-prediction/blob/main/1_data_wrangling_and_cleaning.ipynb): Data preprocessing and cleaning
- [2_eda.ipynb](https://github.com/MeiChieh/lending-club-loan-grade-prediction/blob/main/2_eda.ipynb): Exploratory data analysis
- [3_modeling.ipynb](https://github.com/MeiChieh/lending-club-loan-grade-prediction/blob/main/3_modeling): Model development and evaluation

## 📈 Analysis Structure

### 1. Data Cleaning

- Detect null values, duplicates, and anomalies
- Data wrangling
- Select pre-loan features
- Use NLTK + MultinomialNB to process loan titles and classify them into 14 groups

### 2. EDA

- Feature and target distribution analysis
- Feature correlation studies
- Statistical testing
- Feature selection with Boruta

### 3. Model Development

- Handling imbalanced datasets (SMOTE/BalancedRandomForest)
- AutoML: Optuna for hyperparameter tuning
- Model selection
- Feature importance analysis

## 📁 Project Structure

```
├── data/ # Dataset directory
├── 1_data_wrangling_and_cleaning.ipynb
├── 2_eda.ipynb
├── 3_modeling.ipynb
├── lending_club_deployment/ # Deployment application
├── deployed_app_prediction_demo.ipynb
├── requirements.txt
└── README.md
```

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Flask

## 🚀 Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from [source](https://storage.googleapis.com/335-lending-club/lending-club.zip)
4. Place the dataset files in the `data/` directory

## 💡 Usage

For local development:

1. Install dependencies from requirements.txt
2. Download dataset from the provided source
3. Place files under data/ in project root


## 🔄 Future Improvements

1. Implement additional feature engineering techniques
2. Explore more advanced machine learning models
3. Add real-time monitoring for model performance
4. Develop a user-friendly web interface
5. Implement model versioning and A/B testing

## 📦 Dependencies

Key dependencies include:

- numpy
- pandas
- scikit-learn
- nltk
- flask
- docker

For a complete list, see `requirements.txt`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Mei-Chieh Chien
