Network Anomaly Detection Dashboard

Project Overview

The UGR'16 Network Anomaly Detection Dashboard is a professional-grade web application developed using Python and Dash, designed to analyze and visualize network traffic data from the UGR'16 dataset. Hosted on GitHub, this project leverages an ensemble of machine learning models—XGBoost, Isolation Forest, and One-Class SVM—to detect network anomalies in real-time, delivering comprehensive insights through interactive visualizations and detailed performance metrics.

Key Features

Real-Time Data Simulation: Emulates network traffic with configurable update intervals for continuous monitoring.

Advanced Machine Learning Models:

XGBoost: Gradient boosting classifier for multi-class anomaly detection.

Isolation Forest: Unsupervised anomaly detection with contamination tuning.

One-Class SVM: Linear kernel for efficient anomaly identification.

Ensemble: Majority voting for robust predictions.

Interactive Visualizations:

Time-series charts for network traffic patterns.

Histograms for feature distribution of normal vs. anomalous traffic.

Confusion matrix visualizations for detection performance.

Ensemble voting analysis with model agreement heatmaps.

Alert System: Displays recent anomalies with metadata, including protocol, ports, and severity scores.

Performance Metrics: Tracks accuracy, specificity, attack detection rate, and false positive rate.

User Interface: Features model selection, feature exploration, and adjustable update intervals (1-10 seconds).

Prerequisites

Python: Version 3.8 or higher.
Dependencies: Install required packages using:
pip install dash dash-bootstrap-components pandas numpy xgboost scikit-learn plotly
Dataset: The sample_july_final.pkl file from the UGR'16 dataset, expected at D:/Network anomoly detection/.
Hardware: Recommended 16GB RAM and a multi-core processor for model training.
Install dependencies:
pip install -r requirements.txt
Ensure the sample_july_final.pkl dataset is placed at D:/Network anomoly detection/. Update the path in Dashboard.py if using a different location.
Run the application:
python Dashboard.py
Access the dashboard via a web browser at http://127.0.0.1:8050.
Usage Instructions
Start/Stop Monitoring: Use the "Start Monitoring" button in the navbar to initiate data simulation. Toggle to "Stop Monitoring" to pause.

Model Selection: Select a model (XGBoost, Isolation Forest, One-Class SVM, or Ensemble) via the control panel dropdown.
Feature Analysis: Choose a feature (e.g., packets, bytes, duration, source/destination port) to visualize trends and distributions.
Update Interval: Adjust the refresh rate (1-10 seconds) using the slider for real-time updates
Alerts: View detected anomalies in the "Recent Alerts" card, detailing protocol, ports, and contributing models.
Metrics Monitoring: Review key performance indicators (accuracy, specificity, attack detection rate, false positive rate) in the stats cards.
Repository Structure
Dashboard.py: Core application script handling data preprocessing, model training, simulation, and Dash interface.
requirements.txt: Lists Python dependencies for easy setup.
README.md: This documentation file.

Note: The sample_july_final.pkl dataset is not included in the repository due to its size and licensing. Users must source it separately.
Dataset Details
The UGR'16 dataset (sample_july_final.pkl) contains network traffic records with:
Numerical Features: dateTime, duration, packets, bytes, tos.
Categorical Features: protocol, flag, srcPort, dstPort.
Label: label (background for normal traffic, other values for anomalies).
Machine Learning Models
XGBoost: Trained on 2,000,000 samples with a max depth of 6, learning rate of 0.1, and 100 estimators.
Isolation Forest: Trained on 300,000 samples with 2% contamination and a random state of 3.
One-Class SVM: Trained on 20,000 samples using a linear kernel for computational efficiency.
Ensemble: Combines predictions via majority voting (threshold: 0.5).

Limitations
The dataset path is hardcoded to D:/Network anomoly detection/sample_july_final.pkl. Update Dashboard.py for alternative paths.
Model training is resource-intensive, particularly for XGBoost and Isolation Forest.
The One-Class SVM uses a smaller training subset, which may impact its performance.
The application simulates real-time data using a shuffled dataset, not live network feeds.
