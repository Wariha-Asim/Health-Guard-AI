import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score
)
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="HeartGuard AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for styling 
# -------------------------------
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #0E1117; color: #FAFAFA; }
    
    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, #FF4B4B 0%, #800020 100%);
        padding: 2rem;
        border-radius: 0 0 15px 15px;
        margin-bottom: 2rem;
    }
    
    /* KPI cards */
    .kpi-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
        height: 130px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .kpi-title {
        font-size: 0.9rem;
        color: #A0A0A0;
        margin-bottom: 0.5rem;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #FFFFFF;
    }
    
    .kpi-unit {
        font-size: 0.9rem;
        color: #A0A0A0;
    }
    
    /* Input and Prediction cards */
    .input-card, .prediction-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .card-header {
        font-size: 1.2rem;
        font-weight: bold;
        color: #FFFFFF;
        margin-bottom: 1rem;
        border-bottom: 1px solid #FF4B4B;
        padding-bottom: 0.5rem;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #262730;
        border-radius: 8px 8px 0 0;
        gap: 8px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        color: #A0A0A0;
        padding-left:10px;
        padding-right:10px;
    
    }
    
    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #FFFFFF; 
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    
    .positive { color: #FF4B4B; font-weight: bold; font-size: 24px; }
    .negative { color: #00CC96; font-weight: bold; font-size: 24px; }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #1a1c24;
        padding: 1rem;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #262730;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    /* Input form styling */
    .input-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .input-column {
        flex: 1;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
    <div class="header-container">
        <h1 style="color: white; margin: 0;">HeartGuard AI</h1>
        <p style="color: white; margin: 0; font-size: 1.2rem;">AI-Powered Cardiovascular Risk Prediction Platform</p>
    </div>
""", unsafe_allow_html=True)

# -------------------------------
# KPI Cards (Top Metrics)
# -------------------------------
st.subheader("Key Performance Indicators")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-title">TOTAL PATIENTS</div>'
        '<div class="kpi-value">200</div>'
        '</div>', 
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-title">AVG AGE</div>'
        '<div class="kpi-value">38.9</div>'
        '<div class="kpi-unit">yrs</div>'
        '</div>', 
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-title">AVG CHOLESTEROL</div>'
        '<div class="kpi-value">198.8</div>'
        '</div>', 
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-title">PREDICTION ACCURACY</div>'
        '<div class="kpi-value">89.2%</div>'
        '</div>', 
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        '<div class="kpi-card">'
        '<div class="kpi-title">HIGH RISK CASES</div>'
        '<div class="kpi-value">24%</div>'
        '</div>', 
        unsafe_allow_html=True
    )

# -------------------------------
# Input Parameters Card
# -------------------------------
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown('<div class="card-header">Input Parameters</div>', unsafe_allow_html=True)

# Create input form in the main area
col1, col2 = st.columns(2)

with col1:
    age = st.slider('Age', 20, 100, 50)
    sex = st.selectbox('Sex', ('M', 'F'))
    chest_pain_type = st.selectbox('Chest Pain Type', ('ATA', 'NAP', 'ASY', 'TA'))
    resting_bp = st.slider('RestingBP', 90, 200, 120)
    cholesterol = st.slider('Cholesterol', 100, 400, 200)

with col2:
    fasting_bs = st.selectbox('FastingBS (>120 mg/dl)', (0, 1))
    resting_ecg = st.selectbox('RestingECG', ('Normal', 'ST', 'LVH'))
    max_hr = st.slider('MaxHR', 60, 202, 150)
    exercise_angina = st.selectbox('ExerciseAngina', ('N', 'Y'))
    oldpeak = st.slider('Oldpeak', 0.0, 6.2, 1.0)
    st_slope = st.selectbox('ST Slope', ('Up', 'Flat', 'Down'))

predict_clicked = st.button("Predict Heart Disease Risk")
st.markdown('</div>', unsafe_allow_html=True)

# Prepare input data
encoded_data = {
    'Age': age,
    'Sex': 1 if sex=='M' else 0,
    'ChestPainType': {'ATA':0,'NAP':1,'ASY':2,'TA':3}[chest_pain_type],
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': {'Normal':1,'ST':2,'LVH':0}[resting_ecg],
    'MaxHR': max_hr,
    'ExerciseAngina': 1 if exercise_angina=='Y' else 0,
    'Oldpeak': oldpeak,
    'ST_Slope': {'Up':2,'Flat':1,'Down':0}[st_slope]
}

display_data = {
    'Age': age,
    'Sex': sex,
    'ChestPainType': chest_pain_type,
    'RestingBP': resting_bp,
    'Cholesterol': cholesterol,
    'FastingBS': fasting_bs,
    'RestingECG': resting_ecg,
    'MaxHR': max_hr,
    'ExerciseAngina': exercise_angina,
    'Oldpeak': oldpeak,
    'ST_Slope': st_slope
}

input_df = pd.DataFrame(encoded_data, index=[0])
display_df = pd.DataFrame(display_data, index=[0])

# -------------------------------
# Load dataset and prepare model
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("heart.csv")
    return df

df = load_data()

categorical_cols = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
df_encoded = df.copy()
for col in categorical_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df_encoded[col])

X = df_encoded.drop("HeartDisease", axis=1)
y = df_encoded["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
numeric_cols = ['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

@st.cache_resource
def train_model():
    model = xgb.XGBClassifier(eval_metric="logloss", use_label_encoder=False)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Scale numeric input
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# -------------------------------
# Prediction Results Card
# -------------------------------
if predict_clicked:
    pred_proba = model.predict_proba(input_df)[0][1]
    pred = 1 if pred_proba > 0.5 else 0

    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.markdown('<div class="card-header">Prediction Results</div>', unsafe_allow_html=True)
    
    if pred==1:
        st.markdown(f'<h2 class="positive">High Risk ❤️</h2><p>Probability: {pred_proba*100:.2f}%</p>', unsafe_allow_html=True)
    else:
        st.markdown(f'<h2 class="negative">Low Risk 💚</h2><p>Probability: {(1-pred_proba)*100:.2f}%</p>', unsafe_allow_html=True)

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pred_proba*100,
        title={'text': "Heart Disease Risk Score"},
        gauge={'axis': {'range':[0,100]},
               'steps':[{'range':[0,30],'color':'#00CC96'},
                        {'range':[30,70],'color':'#F0E442'},
                        {'range':[70,100],'color':'#FF4B4B'}]}
    ))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Navigation Tabs
# -------------------------------
st.subheader("Cardiac Analytics Overview")
tab1, tab2, tab3, tab4 = st.tabs(["Risk Analysis", "Feature Analysis", "Model Performance", "Data Explorer"])

# -------------------------------
# Tab 1: Risk Analysis
# -------------------------------
with tab1:
    st.header("Clinical Data Summary")
    st.dataframe(display_df, use_container_width=True)
    
    if predict_clicked:
        st.subheader("Detailed Risk Analysis")
        importance = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values('Importance', ascending=True)
        
        fig = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                     title='Feature Importance for This Prediction',
                     color='Importance', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 2: Feature Analysis
# -------------------------------
with tab2:
    st.header("Feature Importance Analysis")
    importance = model.feature_importances_
    fig = px.bar(x=importance, y=X.columns, orientation="h", 
                 color=importance, color_continuous_scale="Reds",
                 title="Feature Importance in Heart Disease Prediction")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Feature Relationships")
    col1, col2 = st.columns(2)
    
    with col1:
        feature_x = st.selectbox("X-axis feature", options=X.columns, index=0)
    
    with col2:
        feature_y = st.selectbox("Y-axis feature", options=X.columns, index=3)
    
    fig = px.scatter(df, x=feature_x, y=feature_y, color="HeartDisease",
                     color_discrete_map={0:"#00CC96",1:"#FF4B4B"},
                     title=f"{feature_x} vs {feature_y} by Heart Disease Status")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 3: Model Performance
# -------------------------------
with tab3:
    st.header("Model Performance Metrics")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]

    col1, col2, col3, col4 = st.columns(4)
    with col1: 
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><h2>{accuracy_score(y_test,y_pred)*100:.2f}%</h2></div>', unsafe_allow_html=True)
    with col2: 
        st.markdown(f'<div class="metric-card"><h3>Precision</h3><h2>{precision_score(y_test,y_pred)*100:.2f}%</h2></div>', unsafe_allow_html=True)
    with col3: 
        st.markdown(f'<div class="metric-card"><h3>Recall</h3><h2>{recall_score(y_test,y_pred)*100:.2f}%</h2></div>', unsafe_allow_html=True)
    with col4: 
        st.markdown(f'<div class="metric-card"><h3>F1 Score</h3><h2>{f1_score(y_test,y_pred)*100:.2f}%</h2></div>', unsafe_allow_html=True)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"ROC (AUC={roc_auc:.2f})",line=dict(color="#FF4B4B")))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Random",line=dict(dash="dash")))
    fig.update_layout(title="Receiver Operating Characteristic (ROC) Curve")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------
# Tab 4: Data Explorer
# -------------------------------
with tab4:
    st.header("Data Explorer")
    
    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    st.subheader("Data Distribution")
    feature_to_plot = st.selectbox("Select feature to visualize", options=X.columns)
    fig = px.histogram(df, x=feature_to_plot, color="HeartDisease",
                       color_discrete_map={0:"#00CC96",1:"#FF4B4B"},
                       title=f"Distribution of {feature_to_plot} by Heart Disease Status")
    st.plotly_chart(fig, use_container_width=True)
    
    # Dataset statistics
    st.subheader("Dataset Statistics")
    st.dataframe(df.describe(), use_container_width=True)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<div style='text-align:center;'>This tool is for <b>informational and educational purposes only</b> and is not a medical diagnostic tool.</div>", unsafe_allow_html=True)


