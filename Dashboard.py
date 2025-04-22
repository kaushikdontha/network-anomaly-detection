import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import queue
import time
from collections import deque, defaultdict
import xgboost as xgb
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import Normalizer, LabelEncoder
from plotly.subplots import make_subplots
import plotly.express as px
from dash.exceptions import PreventUpdate
import warnings
warnings.filterwarnings('ignore')

# Load dataset
print("Loading dataset...")
path = "D:/Network anomoly detection/sample_july_final.pkl"
dataset = pd.read_pickle(path)


def prepare_data(dataset):
    X_numerical = dataset[["dateTime", "duration", "packets", "bytes", "tos"]].copy()
    X_numcat = dataset[["dstPort", "srcPort"]].copy()
    X_categorical = dataset[["protocol", "flag"]].copy()
    
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(dataset.label)
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)
    
    X_num_norm = pd.DataFrame(Normalizer().fit_transform(X_numerical), columns=X_numerical.columns)
    
    def create_range_port(val):
        if val <= 500: return "a"
        elif val <= 1000: return "b"
        elif val <= 2000: return "c"
        elif val <= 3000: return "e"
        elif val <= 6500: return "f"
        elif val <= 20_000: return "g"
        elif val <= 32_000: return "h"
        return "i"
    
    X_numcat_cat = X_numcat.srcPort.apply(create_range_port).to_frame()
    X_numcat_cat['dstPort'] = X_numcat.dstPort.apply(create_range_port)
    X_numcat_onehot = pd.get_dummies(X_numcat_cat)
    
    X_cat_onehot = pd.get_dummies(X_categorical)
    X_feature = pd.concat([X_num_norm, X_cat_onehot, X_numcat_onehot], axis=1)
    
    return X_feature, Y, label_encoder

def plot_metrics(y_pred, y_true):
    metrics = {}
    background_class = label_encoder.transform(['background'])[0]
    
    if isinstance(y_pred[0], (int, np.integer)):  # XGBoost
        y_pred_binary = np.where(y_pred == background_class, 0, 1)
    else:  # Isolation Forest and SVM
        y_pred_binary = np.where(y_pred == -1, 1, 0)
    
    y_true_binary = np.where(y_true == background_class, 0, 1)
    
    nb_values = len(y_true_binary)
    nb_attacks = y_true_binary.sum()
    nb_normal = nb_values - nb_attacks
    
    correct = (y_pred_binary == y_true_binary).sum()
    correct_normal = np.where((y_pred_binary == 0) & (y_true_binary == 0), 1, 0).sum()
    correct_attack = np.where((y_pred_binary == 1) & (y_true_binary == 1), 1, 0).sum()
    false_positive = np.where((y_pred_binary == 1) & (y_true_binary == 0), 1, 0).sum()
    
    metrics.update({
        'accuracy': round(correct / nb_values * 100, 2),
        'specificity': round(correct_normal / nb_normal * 100, 2) if nb_normal > 0 else 0,
        'attack_accuracy': round(correct_attack / nb_attacks * 100, 2) if nb_attacks > 0 else 0,
        'false_positive_rate': round(false_positive / (correct_normal + false_positive) * 100, 2) if (correct_normal + false_positive) > 0 else 0
    })
    return metrics


def ensemble_predict(features, threshold=0.5):
    xgb_pred = xgb_model.predict(features)[0]
    iso_pred = iso_forest.predict(features)[0]
    svm_pred = one_class_svm.predict(features)[0]
    
    background_class = label_encoder.transform(['background'])[0]
    xgb_binary = 0 if xgb_pred == background_class else 1
    iso_binary = 0 if iso_pred == 1 else 1
    svm_binary = 0 if svm_pred == 1 else 1
    
    vote_percentage = (xgb_binary + iso_binary + svm_binary) / 3
    return 'anomaly' if vote_percentage >= threshold else 'normal'

# Prepare data
print("Preparing data...")
X_feature, Y, label_encoder = prepare_data(dataset)

# Train models
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(
    max_depth=6, learning_rate=0.1, n_estimators=100, objective='multi:softprob',
    num_class=len(np.unique(Y)), n_jobs=6, verbosity=0
)
xgb_model.fit(X_feature[:2_000_000], Y[:2_000_000])

print("Training Isolation Forest...")
iso_forest = IsolationForest(random_state=3, contamination=0.02, n_jobs=6)
iso_forest.fit(X_feature[:300_000])

print("Training One-Class SVM...")
one_class_svm = OneClassSVM(kernel="linear")
one_class_svm.fit(X_feature[:20_000])

# Calculate initial metrics
print("Calculating initial metrics...")
initial_metrics = {
    'xgboost': plot_metrics(xgb_model.predict(X_feature[:100_000]), Y[:100_000]),
    'isolation_forest': plot_metrics(iso_forest.predict(X_feature[:100_000]), Y[:100_000]),
    'svm': plot_metrics(one_class_svm.predict(X_feature[:100_000]), Y[:100_000])
}

# Data simulator class
class DataSimulator:
    def __init__(self, dataset, label_encoder, interval=0.5):
        self.dataset = dataset.sample(frac=1).reset_index(drop=True)  # Shuffle for variety
        self.label_encoder = label_encoder
        self.interval = interval
        self.current_index = 0
        self.running = False
        self.background_class = label_encoder.transform(['background'])[0]
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._simulate_data)
            self.thread.daemon = True
            self.thread.start()
    
    def stop(self):
        self.running = False
    
    def _simulate_data(self):
        while self.running:
            if self.current_index >= len(self.dataset):
                self.current_index = 0
            data_point = self.dataset.iloc[self.current_index].copy()
            data_point['timestamp'] = datetime.now().timestamp()
            
            features = X_feature.iloc[[self.current_index]]
            xgb_pred = xgb_model.predict(features)[0]
            iso_pred = iso_forest.predict(features)[0]
            svm_pred = one_class_svm.predict(features)[0]
            
            xgb_binary = 0 if xgb_pred == self.background_class else 1
            iso_binary = 0 if iso_pred == 1 else 1
            svm_binary = 0 if svm_pred == 1 else 1
            
            ensemble_vote = (xgb_binary + iso_binary + svm_binary) / 3
            ensemble_result = 'anomaly' if ensemble_vote >= 0.5 else 'normal'
            
            predictions = {
                'xgboost': 'normal' if xgb_binary == 0 else 'anomaly',
                'isolation_forest': 'normal' if iso_binary == 0 else 'anomaly',
                'svm': 'normal' if svm_binary == 0 else 'anomaly',
                'ensemble': ensemble_result
            }
            data_point['predictions'] = predictions
            
            try:
                data_queue.put(data_point, block=False)
                recent_data.append(data_point)
                if ensemble_result == 'anomaly':
                    detected_anomalies.append(data_point)
            except queue.Full:
                data_queue.get()
                data_queue.put(data_point)
            
            self.current_index += 1
            time.sleep(self.interval)

# Initialize data structures
data_queue = queue.Queue(maxsize=1000)
recent_data = deque(maxlen=10000)
detected_anomalies = deque(maxlen=1000)
simulator = DataSimulator(dataset, label_encoder)

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "UGR'16 Network Anomaly Detection"

# Layout components (unchanged except for metric label adjustments)
header = dbc.Navbar(
    dbc.Container([
        html.H3("UGR'16 Network Anomaly Detection Dashboard", className="text-white"),
        dbc.Button("Start Monitoring", id="start-button", color="success", className="ml-auto")
    ]),
    color="primary", dark=True
)

control_panel = dbc.Card([
    dbc.CardHeader("Control Panel"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Label("Select Model"),
                dcc.Dropdown(id='model-selector', options=[
                    {'label': 'XGBoost', 'value': 'xgboost'},
                    {'label': 'Isolation Forest', 'value': 'isolation_forest'},
                    {'label': 'One-Class SVM', 'value': 'svm'},
                    {'label': 'Ensemble (Majority Vote)', 'value': 'ensemble'}
                ], value='ensemble', clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Feature Selection"),
                dcc.Dropdown(id='feature-selector', options=[
                    {'label': 'Packets', 'value': 'packets'},
                    {'label': 'Bytes', 'value': 'bytes'},
                    {'label': 'Duration', 'value': 'duration'},
                    {'label': 'Source Port', 'value': 'srcPort'},
                    {'label': 'Destination Port', 'value': 'dstPort'}
                ], value='packets', clearable=False)
            ], width=4),
            dbc.Col([
                html.Label("Update Interval (seconds)"),
                dcc.Slider(id='update-interval', min=1, max=10, value=2, marks={i: str(i) for i in range(1, 11)}, step=1)
            ], width=4)
        ])
    ])
])

# Adjusted labels for clarity
stats_cards = dbc.Row([
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("Overall Accuracy", className="card-title"), html.H2(id="accuracy-metric")])]), width=3),
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("Specificity (Normal)", className="card-title"), html.H2(id="normal-accuracy-metric")])]), width=3),
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("Attack Detection Rate", className="card-title"), html.H2(id="attack-accuracy-metric")])]), width=3),
    dbc.Col(dbc.Card([dbc.CardBody([html.H4("False Positive Rate", className="card-title"), html.H2(id="fpr-metric")])]), width=3),
], id="stats-cards", className="mb-4")

traffic_pattern_card = dbc.Card([dbc.CardHeader("Network Traffic Pattern"), dbc.CardBody([dcc.Graph(id="traffic-pattern")])])
feature_distribution_card = dbc.Card([dbc.CardHeader("Feature Distribution"), dbc.CardBody([dcc.Graph(id="feature-distribution")])])
detection_results_card = dbc.Card([dbc.CardHeader("Detection Results"), dbc.CardBody([dcc.Graph(id="detection-results")])])
ensemble_card = dbc.Card([dbc.CardHeader("Ensemble Voting Results"), dbc.CardBody([dcc.Graph(id="ensemble-results")])])
alerts_card = dbc.Card([dbc.CardHeader("Recent Alerts"), dbc.CardBody([html.Div(id="alerts-container")], style={"maxHeight": "400px", "overflow": "auto"})])

notification_area = html.Div([
    dbc.Toast(id="live-alert-toast", header="Anomaly Detected!", is_open=False, dismissable=True, duration=4000, icon="danger",
              style={"position": "fixed", "top": 66, "right": 10, "width": 350, "zIndex": 1000})
])
notification_store = dcc.Store(id='notification-store', data={'alerts': []})
update_stats = dcc.Interval(id='update-stats', interval=1000, n_intervals=0)
update_graphs = dcc.Interval(id='update-graphs', interval=2000, n_intervals=0)
interval_component = dcc.Interval(id='interval-component', interval=1000, n_intervals=0)

app.layout = html.Div([
    header, notification_area, notification_store, interval_component, update_stats, update_graphs,
    dbc.Container([
        dbc.Row([dbc.Col(control_panel, className="mb-4")]),
        stats_cards,
        dbc.Row([dbc.Col(traffic_pattern_card, width=12, className="mb-4")]),
        dbc.Row([dbc.Col(feature_distribution_card, width=6, className="mb-4"), dbc.Col(detection_results_card, width=6, className="mb-4")]),
        dbc.Row([dbc.Col(ensemble_card, width=6, className="mb-4"), dbc.Col(alerts_card, width=6, className="mb-4")])
    ], fluid=True)
])

# Callbacks
@app.callback(
    [Output("start-button", "children"), Output("start-button", "color")],
    [Input("start-button", "n_clicks")], [State("start-button", "children")]
)
def toggle_simulation(n_clicks, button_text):
    if n_clicks is None:
        return "Start Monitoring", "success"
    if button_text == "Start Monitoring":
        simulator.start()
        return "Stop Monitoring", "danger"
    else:
        simulator.stop()
        return "Start Monitoring", "success"

@app.callback(
    [Output("accuracy-metric", "children"), Output("normal-accuracy-metric", "children"),
     Output("attack-accuracy-metric", "children"), Output("fpr-metric", "children")],
    [Input("update-stats", "n_intervals"), Input("model-selector", "value")]
)
def update_metrics(n, selected_model):
    if not recent_data:
        metrics = initial_metrics.get(selected_model, {'accuracy': 0, 'specificity': 0, 'attack_accuracy': 0, 'false_positive_rate': 0})
    else:
        df = pd.DataFrame(list(recent_data))
        if selected_model == 'ensemble':
            df['prediction'] = df['predictions'].apply(lambda x: x['ensemble'])
        else:
            df['prediction'] = df['predictions'].apply(lambda x: x[selected_model])
        
        df['pred_binary'] = df['prediction'].apply(lambda x: 1 if x == 'anomaly' else 0)
        df['true_binary'] = df['label'].apply(lambda x: 0 if x == 'background' else 1)
        
        total = len(df)
        correct = (df['pred_binary'] == df['true_binary']).sum()
        normal_total = (df['true_binary'] == 0).sum()  # TN + FP
        normal_correct = ((df['pred_binary'] == 0) & (df['true_binary'] == 0)).sum()  # TN
        attack_total = (df['true_binary'] == 1).sum()  # TP + FN
        attack_correct = ((df['pred_binary'] == 1) & (df['true_binary'] == 1)).sum()  # TP
        false_positives = ((df['pred_binary'] == 1) & (df['true_binary'] == 0)).sum()  # FP
        
        metrics = {
            'accuracy': round(correct / total * 100, 2) if total > 0 else 0,
            'specificity': round(normal_correct / normal_total * 100, 2) if normal_total > 0 else 0,
            'attack_accuracy': round(attack_correct / attack_total * 100, 2) if attack_total > 0 else 0,
            'false_positive_rate': round(false_positives / (normal_correct + false_positives) * 100, 2) if (normal_correct + false_positives) > 0 else 0
        }
    return f"{metrics['accuracy']}%", f"{metrics['specificity']}%", f"{metrics['attack_accuracy']}%", f"{metrics['false_positive_rate']}%"

@app.callback(
    Output("traffic-pattern", "figure"),
    [Input("update-graphs", "n_intervals"), Input("feature-selector", "value")]
)
def update_traffic_pattern(n, selected_feature):
    if not recent_data:
        fig = go.Figure()
        fig.update_layout(title="Network Traffic Pattern (No Data)", xaxis_title="Time", yaxis_title="Count", template="plotly_white", height=400)
        return fig
    
    df = pd.DataFrame(list(recent_data))
    if 'timestamp' not in df.columns:
        df['timestamp'] = [datetime.now().timestamp() - i*0.5 for i in range(len(df))]
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.sort_values('time')
    
    df['anomaly'] = df['predictions'].apply(lambda x: any(pred == 'anomaly' for pred in x.values()))
    now = datetime.now()
    time_bins = [(now - timedelta(seconds=i)).timestamp() for i in range(0, 61, 5)]
    time_bins.reverse()
    df['time_bin'] = pd.cut(df['timestamp'], bins=time_bins, labels=[pd.to_datetime(t, unit='s').strftime('%H:%M:%S') for t in time_bins[:-1]])
    
    traffic_counts = df.groupby(['time_bin', 'anomaly']).size().unstack(fill_value=0)
    if True not in traffic_counts.columns:
        traffic_counts[True] = 0
    if False not in traffic_counts.columns:
        traffic_counts[False] = 0
    traffic_counts = traffic_counts.reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Bar(x=traffic_counts['time_bin'], y=traffic_counts[False], name='Normal Traffic', marker_color='#2ecc71'))
    fig.add_trace(go.Bar(x=traffic_counts['time_bin'], y=traffic_counts[True], name='Anomalous Traffic', marker_color='#e74c3c'))
    
    if selected_feature in df.columns:
        feature_data = df.groupby('time_bin')[selected_feature].mean().reset_index()
        fig.add_trace(go.Scatter(x=feature_data['time_bin'], y=feature_data[selected_feature], name=f'{selected_feature.capitalize()} (avg)', line=dict(color='#3498db', width=3), yaxis='y2'))
        fig.update_layout(yaxis2=dict(title=f"{selected_feature.capitalize()}", overlaying="y", side="right"))
    
    fig.update_layout(title="Network Traffic Pattern", xaxis_title="Time", yaxis_title="Packet Count", template="plotly_white", barmode='stack', height=400,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

# Remaining callbacks (unchanged for brevity but validated)
@app.callback(
    Output("feature-distribution", "figure"),
    [Input("update-graphs", "n_intervals"), Input("feature-selector", "value")]
)
def update_feature_distribution(n, selected_feature):
    if not recent_data or selected_feature not in recent_data[0]:
        fig = go.Figure()
        fig.update_layout(title=f"{selected_feature.capitalize()} Distribution (No Data)", xaxis_title=selected_feature.capitalize(), yaxis_title="Count", template="plotly_white", height=400)
        return fig
    
    df = pd.DataFrame(list(recent_data))
    df['anomaly'] = df['predictions'].apply(lambda x: any(pred == 'anomaly' for pred in x.values()))
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=df[df['anomaly'] == False][selected_feature], name='Normal Traffic', marker_color='#2ecc71', opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=df[df['anomaly'] == True][selected_feature], name='Anomalous Traffic', marker_color='#e74c3c', opacity=0.7, nbinsx=30))
    fig.update_layout(title=f"{selected_feature.capitalize()} Distribution", xaxis_title=selected_feature.capitalize(), yaxis_title="Count", template="plotly_white", barmode='overlay', height=400)
    return fig

@app.callback(
    Output("detection-results", "figure"),
    [Input("update-graphs", "n_intervals"), Input("model-selector", "value")]
)
def update_detection_results(n, selected_model):
    if not recent_data:
        fig = go.Figure()
        fig.update_layout(title="Detection Results (No Data)", xaxis_title="Category", yaxis_title="Count", template="plotly_white", height=400)
        return fig
    
    df = pd.DataFrame(list(recent_data))
    df['prediction'] = df['predictions'].apply(lambda x: x[selected_model])
    results = {
        'True Negative': len(df[(df['label'] == 'background') & (df['prediction'] == 'normal')]),
        'False Positive': len(df[(df['label'] == 'background') & (df['prediction'] == 'anomaly')]),
        'False Negative': len(df[(df['label'] != 'background') & (df['prediction'] == 'normal')]),
        'True Positive': len(df[(df['label'] != 'background') & (df['prediction'] == 'anomaly')])
    }
    fig = go.Figure(data=[go.Bar(x=list(results.keys()), y=list(results.values()), marker_color=['#2ecc71', '#e74c3c', '#f1c40f', '#3498db'])])
    fig.update_layout(title=f"Detection Results ({selected_model.capitalize()})", xaxis_title="Category", yaxis_title="Count", template="plotly_white", height=400)
    return fig

@app.callback(
    Output("ensemble-results", "figure"),
    [Input("update-graphs", "n_intervals")]
)
def update_ensemble_results(n):
    if not recent_data:
        fig = go.Figure()
        fig.update_layout(title="Ensemble Results (No Data)", template="plotly_white", height=400)
        return fig
    
    df = pd.DataFrame(list(recent_data))
    model_counts = {
        'XGBoost': sum(1 for d in recent_data if d['predictions']['xgboost'] == 'anomaly'),
        'Isolation Forest': sum(1 for d in recent_data if d['predictions']['isolation_forest'] == 'anomaly'),
        'One-Class SVM': sum(1 for d in recent_data if d['predictions']['svm'] == 'anomaly'),
        'Ensemble': sum(1 for d in recent_data if d['predictions']['ensemble'] == 'anomaly')
    }
    agreement_data = [dict(XGBoost=1 if d['predictions']['xgboost'] == 'anomaly' else 0,
                          Isolation_Forest=1 if d['predictions']['isolation_forest'] == 'anomaly' else 0,
                          One_Class_SVM=1 if d['predictions']['svm'] == 'anomaly' else 0)
                     for d in recent_data if d['predictions']['ensemble'] == 'anomaly']
    agreement_df = pd.DataFrame(agreement_data) if agreement_data else pd.DataFrame({'XGBoost': [0], 'Isolation_Forest': [0], 'One_Class_SVM': [0]})
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Model Anomaly Detections", "Model Agreement on Anomalies"), specs=[[{"type": "bar"}, {"type": "heatmap"}]])
    fig.add_trace(go.Bar(x=list(model_counts.keys()), y=list(model_counts.values()), marker_color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6']), row=1, col=1)
    corr = agreement_df.corr()
    fig.add_trace(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, colorscale='Viridis', zmin=-1, zmax=1), row=1, col=2)
    fig.update_layout(template='plotly_white', title='Ensemble Model Analysis', height=400)
    return fig

@app.callback(
    Output("alerts-container", "children"),
    [Input("notification-store", "data")]
)
def update_alerts_container(data):
    if not data or not data['alerts']:
        return html.Div([html.P("No alerts detected", className="text-muted"),
                         html.Img(src="https://img.icons8.com/color/96/000000/ok--v1.png", className="mx-auto d-block mt-3")])
    
    alerts = []
    for alert in reversed(data['alerts'][:10]):
        severity = alert['severity']
        severity_color = "danger" if severity > 0.7 else "warning" if severity > 0.4 else "info"
        alerts.append(dbc.Card([
            dbc.CardHeader([html.Div([html.Span(datetime.fromtimestamp(alert['timestamp']).strftime('%H:%M:%S'), className="float-right text-muted"),
                                     html.H5(f"Anomaly #{alert['id']}", className="mb-0")])], className=f"bg-{severity_color} text-white"),
            dbc.CardBody([dbc.Row([dbc.Col([html.P([html.Strong("Protocol: "), f"{alert['protocol']}"]),
                                           html.P([html.Strong("Source Port: "), f"{alert['src_port']}"]),
                                           html.P([html.Strong("Destination Port: "), f"{alert['dst_port']}"])], width=6),
                                  dbc.Col([html.P([html.Strong("Detected by:")]), html.Ul([html.Li(model) for model in alert['models']], className="pl-3")], width=6)])])
        ], className="mb-3"))
    return alerts

@app.callback(
    [Output('live-alert-toast', 'is_open'), Output('live-alert-toast', 'children'), Output('live-alert-toast', 'header')],
    [Input('notification-store', 'data')], [State('live-alert-toast', 'is_open')]
)
def show_toast(data, is_open):
    if not data or not data['alerts']:
        raise PreventUpdate
    latest_alert = data['alerts'][-1]
    if is_open:
        raise PreventUpdate
    
    alert_time = datetime.fromtimestamp(latest_alert['timestamp']).strftime('%H:%M:%S')
    severity = latest_alert['severity']
    severity_level = "Critical" if severity > 0.7 else "High" if severity > 0.4 else "Medium"
    content = html.Div([
        html.P([html.Strong("Time: "), f"{alert_time}"]), html.P([html.Strong("Protocol: "), f"{latest_alert['protocol']}"]),
        html.P([html.Strong("Source Port: "), f"{latest_alert['src_port']}"]), html.P([html.Strong("Destination Port: "), f"{latest_alert['dst_port']}"]),
        html.P([html.Strong("Detected by: "), ", ".join(latest_alert['models'])]), dbc.Progress(value=severity * 100, color="danger" if severity > 0.7 else "warning" if severity > 0.4 else "info", className="mb-2")
    ])
    return True, content, f"{severity_level} Alert: Anomaly Detected!"

@app.callback(
    Output('notification-store', 'data'),
    [Input('interval-component', 'n_intervals')], [State('notification-store', 'data')]
)
def update_notification_store(n, data):
    if not detected_anomalies or n is None:
        return data if data else {'alerts': []}
    
    current_time = time.time()
    new_alerts = []
    for anomaly in list(detected_anomalies):
        if 'timestamp' in anomaly and current_time - anomaly['timestamp'] < 5:
            alert_data = {
                'id': str(len(data['alerts']) + len(new_alerts) + 1) if data else str(len(new_alerts) + 1),
                'timestamp': anomaly['timestamp'], 'protocol': anomaly.get('protocol', 'Unknown'),
                'src_port': anomaly.get('srcPort', 'Unknown'), 'dst_port': anomaly.get('dstPort', 'Unknown'),
                'models': [model for model, pred in anomaly['predictions'].items() if pred == 'anomaly'],
                'severity': len([pred for pred in anomaly['predictions'].values() if pred == 'anomaly']) / len(anomaly['predictions'])
            }
            new_alerts.append(alert_data)
    
    if not new_alerts:
        return data if data else {'alerts': []}
    updated_data = {'alerts': (data['alerts'] + new_alerts) if data else new_alerts}
    if len(updated_data['alerts']) > 50:
        updated_data['alerts'] = updated_data['alerts'][-50:]
    return updated_data

@app.callback(
    Output("update-graphs", "interval"),
    [Input("update-interval", "value")]
)
def update_interval(value):
    return value * 1000

if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
    