# Problem Statement
The goal of this project is to develop a machine learning-based system to detect network anomalies and classify attack types in network traffic. Network anomalies can significantly impact system performance and security, and timely detection is critical to mitigate risks.

The problem is divided into two sub-tasks:

- Binary Classification: Determine whether a network connection is normal (0) or anomalous (1).
- Multiclass Classification: If a connection is anomalous, classify it into specific attack types such as DOS, Probe, R2L, etc.

# Target Metric
The key evaluation metrics for the models are:

- Accuracy: Overall correctness of predictions.
- Precision, Recall, and F1-score: To measure performance on imbalanced datasets.
- Support: Distribution of classes in predictions to ensure model handles rare attack types effectively.
# Steps Taken
## 1. Exploratory Data Analysis (EDA)
EDA was conducted to understand the data distribution, detect potential outliers, and assess relationships between features and the target variable (attack_or_normal). Key insights include:

Feature Distributions: Most numerical features such as srcbytes and dstbytes showed right-skewed distributions, while categorical features like protocoltype and flag had distinct value distributions.
Class Imbalance: The target variable (attack_or_normal) showed a significant imbalance, with normal connections being more frequent than anomalies.
Feature-Target Relationships:
Anomalies tend to occur with specific flags (REJ, S0) and services (http, ftp_data).
srcbytes and dstbytes exhibited higher variability in anomalous connections.
## 2. Hypothesis Testing
Several hypotheses were tested to derive meaningful insights:

Protocol Type vs. Anomalies: A chi-square test confirmed that anomalies are more likely associated with specific protocols (e.g., tcp and udp).
Traffic Volume: T-tests revealed significant differences in srcbytes and dstbytes between normal and anomalous connections.
Connection Flags: Logistic regression analysis identified specific flags strongly associated with anomalies.
## 3. Feature Engineering
To enhance the dataset for modeling, the following features were engineered:

Derived Features: Speed-related features such as srcbytes/sec and dstbytes/sec were created.
Interaction Terms: Products of error rates and connection counts (serrors_count, rerrors_count) were computed.
Categorical Encoding: Features like protocoltype, service, and flag were encoded using LabelEncoder.
## 4. Machine Learning Modeling
Two models were developed to address the problem:

Binary Classification (Attack or Not):

Model: Decision Tree Classifier
Preprocessing: Encoded categorical features and scaled numerical features using StandardScaler.
Final Accuracy: Achieved an accuracy of 99.87% with an F1-score close to 1.0.
Multiclass Classification (Attack Types):

Model: Decision Tree Classifier
Preprocessing: Similar to binary classification, with training restricted to anomalous data (attack_or_normal = 1).
Final Accuracy: Achieved an accuracy of 97.5% on the test set, with strong performance across all attack categories.
## 5. Deployment
The trained models were deployed using Streamlit, allowing users to:

Input network connection features manually or via CSV upload.
View predictions for binary classification and, if anomalous, the predicted attack type.
Access a user-friendly interface with dropdowns for categorical inputs and auto-scaling for numerical inputs.
Deployment Steps:

Encoders and scalers were saved using pickle to ensure preprocessing consistency during inference.
The Streamlit app was hosted on Streamlit Sharing, providing a globally accessible link.
# Final Scores
## Binary Classification:

- Accuracy: 99.87%
- F1-score: 1.0 for both normal and anomalous classes.
## Multiclass Classification:

- Accuracy: 97.5%
- Precision, Recall, and F1-score were high for major attack categories, with slight degradation for rarer classes.
# Insights and Recommendations
1. Insights:

Features like srcbytes, dstbytes, and error rates (serrorrate, rerrorrate) are critical for distinguishing between normal and anomalous connections.
Anomalous connections often correspond to specific protocols (tcp, udp), flags (REJ, S0), and services (ftp, http).
2. Recommendations:

Real-Time Monitoring: Incorporate the trained models into a real-time monitoring system to detect and classify anomalies dynamically.
Class Imbalance Handling: If rare attack types are critical, consider using SMOTE or other sampling techniques to balance the dataset.
Model Retraining: Periodically retrain the models with updated network traffic data to ensure robustness against evolving attack types.
