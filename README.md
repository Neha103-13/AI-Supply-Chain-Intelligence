# AI Supply Chain Intelligence

An end-to-end AI-powered supply chain analytics and late-delivery risk prediction system built using SQL, Machine Learning, Explainable AI, Flask, and Generative AI.

## Project Overview

This project analyzes supply chain order data to identify delivery patterns, evaluate shipping performance, predict late-delivery risk, and provide explainable insights for business decision-making.

The system combines:

- SQL-based supply chain analytics
- Machine Learning
- XGBoost
- SHAP Explainable AI
- Flask web application
- Generative AI assistant
- Interactive business dashboards

## Key Features

### 1. Supply Chain Analytics

SQL analytics are used to analyze:

- Overall delivery performance
- Late-delivery rates
- Shipping modes
- Markets
- Monthly delivery trends
- Order-level business metrics

### 2. Machine Learning

The project evaluates multiple classification models:

- Majority Baseline
- Logistic Regression
- Random Forest
- XGBoost

A temporal train-test split is used:

- Training: 2015–2017
- Testing: 2018

The target variable is:

`late_delivery_risk`

### 3. Explainable AI

SHAP is used to explain model predictions.

The system identifies:

- Factors pushing predictions toward higher risk
- Factors pushing predictions toward lower risk
- Feature contributions for individual predictions
- Global feature importance

SHAP explanations describe model behavior and should not be interpreted as causal effects.

### 4. Risk Prediction

Users can enter order information through the Flask application and receive:

- Predicted delivery-risk class
- Late-delivery probability
- Risk level
- Positive contributing factors
- Negative contributing factors
- SHAP-based explanation

Risk levels used by the application:

- LOW
- MEDIUM
- HIGH

### 5. Business Intelligence Dashboard

The dashboard provides:

- Total orders
- Late orders
- Overall late-delivery rate
- Average scheduled shipping days
- Market-level performance
- Shipping-mode performance
- Monthly late-delivery trends

### 6. AI Supply Chain Assistant

A Generative AI assistant allows users to ask questions about the project data and prediction results.

Example questions:

- Which shipping mode has the highest late-delivery rate?
- Which market has the highest observed late-delivery rate?
- What is the overall late-delivery rate?
- How do the machine-learning models compare?
- Why was this order predicted as high risk?
- What factors influenced this prediction?
- What business actions can be considered?

The assistant is designed to use project-specific analytics and prediction context.

## Dataset

The project uses the DataCo Supply Chain dataset.

The original dataset contains supply-chain order, customer, product, shipping, sales, and delivery information.

Dataset characteristics:

- 180,519 records
- 53 columns
- Date range: 2015–2018

For machine-learning prediction, the project creates an order-level dataset containing:

- 65,752 orders

## Data Preparation

The original dataset contains fields that are not appropriate for prediction or are not required for the model.

The preprocessing stage includes:

- Data quality inspection
- Duplicate checking
- Missing-value analysis
- Date conversion
- Temporal feature creation
- PII removal
- Leakage-aware feature selection
- Order-level aggregation
- SQL database creation

Sensitive customer information is excluded from the machine-learning features.

Post-event or leakage-prone fields such as actual shipping duration and shipping date are not used as prediction inputs.

## Machine Learning Features

The order-level model uses the following features:

- Order Type
- Market
- Order City
- Order Country
- Order Region
- Order State
- Shipping Mode
- Scheduled Shipping Days
- Customer Segment
- Number of Items
- Number of Different Products
- Total Quantity
- Average Product Price
- Total Discount
- Average Discount Rate
- Order Year
- Order Month
- Order Day of Week

## Model Evaluation Strategy

The project uses a temporal train-test split instead of a random split.

### Training Data

2015–2017

### Test Data

2018

This approach evaluates whether the models can generalize to a later time period.

The test set contains:

- 2,123 orders

The target variable is:

`late_delivery_risk`

## Model Results

Evaluation is performed on the 2018 test set.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Majority Baseline | 56.29% | — | — | — | — |
| Logistic Regression | 69.29% | 80.75% | 59.67% | 68.62% | 75.46% |
| Random Forest | 68.91% | 79.96% | 59.75% | 68.39% | 75.35% |
| XGBoost | 69.19% | 81.64% | 58.41% | 68.10% | 75.40% |

The tested machine-learning models outperform the majority-class baseline on the 2018 test set.

The results page in the application provides a visual comparison of the tested models across multiple evaluation metrics.

## Explainable AI with SHAP

SHAP is integrated into the prediction workflow to explain individual model predictions.

For each prediction, the application can display:

- Positive factors
- Negative factors
- Feature contribution values
- Predicted probability
- Predicted risk level

SHAP values indicate how model features contribute to a particular prediction.

They should not be interpreted as:

- Causal effects
- Percentages of responsibility
- Probabilities
- Independent business impact measurements

The application also groups model-level feature contributions into business-friendly feature categories.

## SQL Analytics

The project uses SQLite for structured supply-chain analytics.

The database contains tables for:

- Customers
- Products
- Orders
- Order Items

SQL analytics are used to calculate:

- Overall late-delivery performance
- Market-level late-delivery rates
- Shipping-mode performance
- Monthly delivery trends
- Order and product metrics

## Business Insights

The application provides business-oriented analysis based on the observed project data.

Examples include:

- Comparison of shipping-mode delivery performance
- Comparison of market-level late-delivery rates
- Monthly late-delivery trends
- Predictive risk scoring
- Explainable model insights

Observed historical patterns are presented as patterns in the dataset and are not treated as proof of causation.

## Technology Stack

### Programming

- Python
- SQL
- HTML
- CSS
- JavaScript

### Machine Learning

- Scikit-learn
- XGBoost
- SHAP

### Data Analysis

- Pandas
- NumPy
- SQLite

### Visualization

- Matplotlib
- Seaborn
- JavaScript-based dashboard visualizations

### Web Application

- Flask

### Generative AI

- OpenRouter
- OpenAI-compatible API

### Development

- Visual Studio Code
- Jupyter Notebook
- Python Virtual Environment

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Neha103-13/AI-Supply-Chain-Intelligence.git
cd AI-Supply-Chain-Intelligence
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the AI Assistant

Create a `.env` file in the project root:

```text
OPENROUTER_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your OpenRouter API key.

The `.env` file must remain local and should not be committed to GitHub.

### 5. Run the Flask Application

From the project root:

```bash
cd app
python app.py
```

### 6. Open the Application

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

The application provides:

- Supply Chain Intelligence Dashboard
- Late-Delivery Risk Prediction
- Machine Learning Model Performance
- Business Insights
- AI Supply Chain Assistant

## Application Pages

### Dashboard

Provides an overview of:

- Total orders
- Late orders
- Late-delivery rate
- Scheduled shipping time
- Market performance
- Shipping-mode performance
- Monthly trends

### Risk Prediction

Allows users to enter order information and receive:

- Late-delivery probability
- Risk level
- Model prediction
- SHAP positive factors
- SHAP negative factors

### Model Performance

Provides:

- Temporal evaluation strategy
- Model comparison
- Accuracy
- Precision
- Recall
- F1 score
- ROC-AUC
- Model comparison visualization

### Business Insights

Provides:

- Dataset overview
- Delivery performance
- Shipping-mode analysis
- Market analysis
- Monthly trends
- Machine-learning insights
- Explainable AI insights
- Business considerations

### AI Supply Chain Assistant

Allows users to ask natural-language questions about:

- Supply-chain statistics
- Shipping modes
- Markets
- Model performance
- Business insights
- Individual predictions
- SHAP explanations

## Project Architecture

```text
                    Supply Chain Dataset
                              |
                              v
                    Data Preprocessing
                              |
               +--------------+--------------+
               |                             |
               v                             v
         SQLite Database               ML Dataset
               |                             |
               v                             v
        SQL Analytics              Feature Engineering
                                             |
                                             v
                                  Machine Learning Models
                                             |
                           +-----------------+----------------+
                           |                 |                |
                           v                 v                v
                    Logistic Regression  Random Forest    XGBoost
                                                               |
                                                               v
                                                        SHAP Explainability
                                                               |
                                                               v
                                                       Flask Application
                                                               |
                     +----------------+----------------+---------+---------+
                     |                |                |                   |
                     v                v                v                   v
                 Dashboard     Risk Prediction   Business Insights   AI Assistant
```

## Project Structure

```text
AI-Supply-Chain-Intelligence/
├── app/
│   ├── app.py
│   ├── predictor.py
│   ├── ai_assistant.py
│   └── templates/
│       ├── index.html
│       ├── dashboard.html
│       ├── model_performance.html
│       ├── business_insights.html
│       └── ai_assistant.html
│
├── data/
│   └── model_results.csv
│
├── models/
│   ├── model_features.pkl
│   └── xgboost_late_delivery_model.pkl
│
├── notebook/
│   └── 01_data_inspection.ipynb
│
├── screenshots/
│   ├── dashboard.png
│   ├── risk_prediction.png
│   ├── model_performance.png
│   ├── business_insights.png
│   └── ai_assistant.png
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt

## Application Screenshots

### Supply Chain Intelligence Dashboard

The dashboard provides an overview of orders, delivery performance, shipping modes, markets, and monthly late-delivery trends.

![Supply Chain Intelligence Dashboard](screenshots/dashboard.png)

### Late-Delivery Risk Prediction

The risk prediction interface uses the trained machine-learning model to estimate late-delivery probability and SHAP to explain the factors contributing to the prediction.

![Late Delivery Risk Prediction](screenshots/risk_prediction.png)

### Machine Learning Model Performance

The model performance page compares the tested classification models using a temporal train-test evaluation strategy.

![Machine Learning Model Performance](screenshots/model_performance.png)

### Business Insights

The business insights page presents supply-chain patterns derived from SQL analytics and machine-learning results.

![Business Insights](screenshots/business_insights.png)

### AI Supply Chain Assistant

The AI assistant allows users to ask natural-language questions about supply-chain analytics and individual model predictions.

![AI Supply Chain Assistant](screenshots/ai_assistant.png)

## Data Privacy

Sensitive customer fields from the original dataset are not used as machine-learning features.

The project excludes personally identifiable information such as:

- Customer Email
- Customer First Name
- Customer Last Name
- Customer Password
- Customer Street
- Customer ZIP Code
- Order ZIP Code

The `.env` file containing the AI API key is also excluded from Git version control.

## Important Machine Learning Notes

### Temporal Evaluation

The model evaluation uses:

- Training period: 2015–2017
- Test period: 2018

This avoids randomly mixing future observations into the training data.

### Prediction Thresholds

The application maps predicted probabilities to application-level risk categories:

- Probability below 0.40 → LOW
- Probability from 0.40 to below 0.70 → MEDIUM
- Probability of 0.70 or higher → HIGH

These thresholds are application-level thresholds and are not presented as validated business decision thresholds.

### SHAP Interpretation

SHAP explains the behavior of the trained model.

A positive SHAP contribution pushes the model toward a higher predicted risk.

A negative SHAP contribution pushes the model toward a lower predicted risk.

SHAP values do not establish causal relationships.

## Future Improvements

Potential future improvements include:

- Model hyperparameter optimization
- Additional temporal and operational features
- Model monitoring
- Automated retraining pipelines
- Cloud deployment
- REST API deployment
- Authentication and role-based access
- Advanced supply-chain forecasting
- Real-time operational data integration
- Automated model monitoring
- Production deployment with containerization

## License

This project is licensed under the MIT License.

## Author

Neha Chougule