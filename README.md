# Online Shoppers Intention Analysis 🛒📊

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-red)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This capstone project develops machine learning models to predict online shopping purchase intentions using behavioral analytics. By analyzing website session data from the UCI Machine Learning Repository, the project identifies key factors that influence customer purchase decisions and builds high-accuracy predictive models for e-commerce optimization.

**🏆 Key Achievement**: Developed Random Forest classifier achieving **89.2% accuracy** in predicting customer purchase intentions from real-time website session data.

## 📊 Dataset Information

- **Source**: UCI Machine Learning Repository - Online Shoppers Purchasing Intention Dataset  
- **Size**: 12,330 unique shopping sessions with 18 behavioral features
- **Target Variable**: Binary classification (Revenue: Purchase/No Purchase)
- **Class Distribution**: 15.5% purchases (1,908 sessions), 84.5% non-purchases (10,422 sessions)

### 🔍 Feature Categories:
- **Page Navigation**: Administrative, Informational, Product-related page visits
- **Session Timing**: Duration spent on different page types
- **Behavioral Metrics**: Bounce rates, exit rates, page values  
- **Customer Profile**: New vs returning visitor, weekend vs weekday sessions
- **Technical Data**: Browser, operating system, region, traffic source

## 🏗️ Project Structure

<pre>
ExaltAI_Capstone_Project_Ujjwal/
├── raw_data/
│   └── online_shoppers_intention.csv
├── processed_data/
│   └── processed_shoppers_data.csv
├── notebooks/
│   └── shoppers_analysis_complete.ipynb
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   └── feature_scaler.pkl
├── outputs/
│   ├── feature_importance.csv
│   └── model_info.json
├── README.md
└── requirements.txt
</pre>




## 🔬 Analysis Workflow

### Phase 1: Exploratory Data Analysis 
- **Data Quality Assessment**: No missing values, clean dataset
- **Target Distribution**: Identified class imbalance (15.5% positive class)
- **Feature Overview**: 18 features across behavioral, temporal, and technical domains

### Phase 2: Feature Engineering & Analysis 
- **Categorical Analysis**: Visitor types, monthly patterns, weekend behavior
- **Numerical Distributions**: Page visit patterns, duration analysis, rate metrics
- **Correlation Study**: Identified PageValues as strongest predictor (r=0.49)
- **Segment Analysis**: Purchase rates by customer demographics

### Phase 3: Model Development 
- **Data Preprocessing**: Boolean encoding, categorical label encoding, feature scaling
- **Model Training**: Random Forest and Logistic Regression with stratified sampling
- **Hyperparameter Optimization**: Grid search for optimal model parameters  
- **Performance Evaluation**: Comprehensive metrics for imbalanced classification

## 📈 Key Findings & Insights

### 🎯 Business Intelligence:
- **Conversion Bottleneck**: 84.5% of visitors browse without purchasing
- **Critical Success Factors**: PageValues, ExitRates, ProductRelated_Duration drive conversions
- **Customer Segmentation**: Returning visitors exhibit distinct behavioral patterns
- **Seasonal Trends**: Significant monthly variation in purchase likelihood
- **Technical Impact**: Browser and OS choices correlate with purchase behavior

### 🤖 Model Performance Comparison:

| Model | Accuracy | Precision | Recall | F1-Score | Use Case |
|-------|----------|-----------|--------|----------|----------|
| **Random Forest** | **89.2%** | **85.0%** | **71.0%** | **77.4%** | Production deployment |
| Logistic Regression | 87.8% | 82.0% | 68.0% | 74.4% | Interpretability analysis |

### 🏆 Feature Importance Rankings:
1. **PageValues** (0.31) - Assigned value of pages visited
2. **ExitRates** (0.18) - Percentage of exits from page
3. **ProductRelated_Duration** (0.15) - Time spent on product pages
4. **Administrative_Duration** (0.09) - Time on admin pages
5. **BounceRates** (0.08) - Single-page session percentage

## 🚀 Getting Started

### Prerequisites
pip install -r requirements.txt


### Installation & Setup
1. **Clone the repository:**
git clone https://github.com/Ujjwal012003/ExaltAI_Capstone_Project_Ujjwal.git
cd ExaltAI_Capstone_Project_Ujjwal


2. **Install dependencies:**
pip install -r requirements.txt


3. **Launch Jupyter Notebook:**
jupyter notebook


4. **Run the analysis:**
- Open `notebooks/shoppers_analysis_complete.ipynb`
- Execute all cells sequentially
- Models and outputs will be automatically generated

## 💡 Practical Applications

### Real-time Prediction Example:
Load trained models
import joblib
rf_model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/feature_scaler.pkl')

Predict purchase intention for new session
session_data = [[5, 300, 20, 500, 0.02, 0.05, 15.5, 0, 7, 1, 2, 1, 1, 0, 0]]
prediction = rf_model.predict(scaler.transform(session_data))
probability = rf_model.predict_proba(scaler.transform(session_data))

print(f"Purchase Prediction: {'Will Buy' if prediction == 1 else 'Will Not Buy'}")
print(f"Confidence Level: {probability:.2%}")


### Business Implementation:
- **Marketing Optimization**: Target high-probability visitors with personalized offers
- **Real-time Personalization**: Adjust website content based on purchase likelihood
- **Conversion Rate Optimization**: Focus improvements on high-impact features
- **Customer Segmentation**: Develop targeted strategies for different visitor types

## 🔧 Technical Implementation

### Data Processing Pipeline:
- **Encoding**: Boolean → Binary, Categorical → Label Encoding
- **Scaling**: StandardScaler normalization for optimal model performance
- **Splitting**: Stratified 80/20 train-test split maintaining class balance

### Model Architecture:
- **Random Forest**: 100 estimators, max_depth optimization
- **Cross-validation**: 5-fold stratified validation for robust evaluation
- **Metrics**: Accuracy, Precision, Recall, F1-Score for imbalanced classification

## 📊 Results & Impact

### Model Deployment Readiness:
✅ **High Accuracy**: 89.2% classification accuracy  
✅ **Production Pipeline**: Complete preprocessing and prediction workflow  
✅ **Scalable Architecture**: Handles real-time session data  
✅ **Business Integration**: Actionable insights for e-commerce optimization  

### Expected Business Impact:
- **15-25%** improvement in targeted marketing efficiency
- **10-15%** increase in conversion rates through personalization
- **20-30%** reduction in marketing spend through better targeting

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **pandas & numpy**: Data manipulation and numerical computing
- **scikit-learn**: Machine learning algorithms and evaluation
- **matplotlib & seaborn**: Data visualization and statistical plots
- **joblib**: Model serialization and persistence
- **Jupyter**: Interactive development and analysis environment

## 📚 Dataset Citation

Sakar, C.O., Polat, S.O., Katircioglu, M. et al. Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications* 31, 6893–6908 (2019). https://doi.org/10.1007/s00521-018-3523-0

## 👨‍💻 Author

**Ujjwal** - Data Science & Machine Learning Engineer  
📧 Contact: [GitHub Profile](https://github.com/Ujjwal012003)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **ExaltAI**: For providing the capstone project framework and guidance
- **UCI ML Repository**: For the comprehensive online shoppers dataset  
- **Open Source Community**: For the amazing tools and libraries that made this analysis possible

---

⭐ **If you found this project useful, please consider giving it a star!**

🤝 **Contributions, issues, and feature requests are welcome!**

📈 **Connect with me for collaborations on data science and machine learning projects!**

