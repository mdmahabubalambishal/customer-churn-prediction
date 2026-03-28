#─── import important libaries ──────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Path Fix ──────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.feature_engineering import engineer_features

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        font-size: 1.3rem;
        font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Load Model & Data ─────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, 'models', 'full_pipeline.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data_path = os.path.join(
        ROOT_DIR, 'data', 'processed', 'cleaned_featured_data.csv'
    )
    return pd.read_csv(data_path)

try:
    pipeline     = load_model()
    df           = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Model/Data load error: {e}")
    st.info(
        "নিশ্চিত করো:\n"
        "- models/full_pipeline.pkl আছে\n"
        "- data/processed/cleaned_featured_data.csv আছে\n"
        "- 03_modeling.ipynb এর Cell 22 run হয়েছে"
    )


# ─── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<h2 style='text-align:center; color:#667eea'>"
        "🔮 Churn Predictor</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔮 Prediction", "📊 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 📌 Model Info")
    st.info(
        "**Algorithm:** XGBoost Tuned\n\n"
        "**ROC-AUC:** ~0.85\n\n"
        "**F1 Score:** ~0.62\n\n"
        "**Dataset:** Telco (7,043 rows)"
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.85rem'>"
        "Built by <b>Mahabub Alam Bishal</b></p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown(
        '<h1 class="main-header">🔮 Customer Churn Predictor</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">'
        'ML-powered system to predict which customers'
        ' are likely to leave'
        '</p>',
        unsafe_allow_html=True
    )

    # ── Metrics Row ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Dataset",    "7,043",   "customers")
    c2.metric("🎯 ROC-AUC",    "0.850",   "+vs baseline")
    c3.metric("⚡ Features",   "28",      "engineered")
    c4.metric("🏆 Best Model", "XGBoost", "tuned")

    st.markdown("---")

    # ── Problem & Solution ──
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Telecom কোম্পানিগুলোর জন্য customer churn
        একটা বড় সমস্যা।

        একজন নতুন customer acquire করতে existing
        customer retain করার চেয়ে **5-7x বেশি খরচ** লাগে।

        এই system টা predict করে **কোন customer
        চলে যেতে পারে** — যাতে আগেই retention
        offer দেওয়া যায়।
        """)

    with col_r:
        st.markdown("### 🛠️ Technical Approach")
        st.markdown("""
        - **EDA:** 15+ visualizations, data quality analysis
        - **Feature Eng.:** 7 নতুন business features তৈরি
        - **Models:** LR, Random Forest, XGBoost, LightGBM
        - **Tracking:** MLflow experiment tracking
        - **Tuning:** RandomizedSearchCV (20 iterations)
        - **Deployment:** Streamlit Cloud
        """)

    st.markdown("---")

    # ── Pipeline Steps ──
    st.markdown("### 🔄 ML Pipeline")
    steps = [
        ("📥", "Raw Data"),
        ("🔍", "EDA"),
        ("⚙️", "Preprocessing"),
        ("🧪", "Feature Eng."),
        ("🤖", "Model Train"),
        ("📊", "Evaluation"),
        ("🚀", "Deployment")
    ]
    cols = st.columns(len(steps))
    for col, (icon, label) in zip(cols, steps):
        with col:
            st.markdown(
                f"<div style='text-align:center; padding:12px 5px;"
                f"background:linear-gradient("
                f"135deg,#667eea20,#764ba220);"
                f"border:1px solid #667eea40;"
                f"border-radius:10px;"
                f"font-size:0.85rem; font-weight:600'>"
                f"{icon}<br>{label}</div>",
                unsafe_allow_html=True
            )

    st.markdown("---")

    # ── Key Findings ──
    st.markdown("### 🔍 Key Business Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.error("📋 **Month-to-month contract**\n\nChurn rate: **42%**")
    with i2:
        st.warning("🌐 **Fiber optic internet**\n\nChurn rate: **42%**")
    with i3:
        st.warning("🕐 **New customers (≤6 mo)**\n\nChurn rate: **47%**")

    i4, i5, i6 = st.columns(3)
    with i4:
        st.success("📅 **Two-year contract**\n\nChurn rate: **3%**")
    with i5:
        st.warning("🔒 **No online security**\n\nChurn rate: **41%**")
    with i6:
        st.success("👥 **Long-term (24+ mo)**\n\nChurn rate: **~15%**")

    st.markdown("---")
    st.success(
        "👈 **Sidebar থেকে 'Prediction' page এ যাও** "
        "— customer data দিয়ে churn predict করো!"
    )


# ══════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.markdown("## 🔮 Predict Customer Churn")
    st.markdown(
        "Customer এর information দাও — model predict করবে।"
    )
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model load হয়নি! Notebook গুলো আগে run করো।")
        st.stop()

    # ── Input Form ──
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Customer Info")
        gender = st.selectbox(
            "Gender", ["Male", "Female"]
        )
        senior_citizen = st.selectbox(
            "Senior Citizen", ["No", "Yes"]
        )
        partner = st.selectbox(
            "Has Partner?", ["No", "Yes"]
        )
        dependents = st.selectbox(
            "Has Dependents?", ["No", "Yes"]
        )
        tenure = st.slider(
            "Tenure (months)", 0, 72, 12
        )

    with col2:
        st.markdown("#### 📱 Services")
        phone_service = st.selectbox(
            "Phone Service", ["Yes", "No"]
        )
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        online_security = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"]
        )
        online_backup = st.selectbox(
            "Online Backup",
            ["No", "Yes", "No internet service"]
        )
        device_protection = st.selectbox(
            "Device Protection",
            ["No", "Yes", "No internet service"]
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"]
        )
        streaming_tv = st.selectbox(
            "Streaming TV",
            ["No", "Yes", "No internet service"]
        )
        streaming_movies = st.selectbox(
            "Streaming Movies",
            ["No", "Yes", "No internet service"]
        )

    with col3:
        st.markdown("#### 💳 Billing")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless_billing = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0, max_value=150.0,
            value=65.0, step=0.5
        )
        # Max value dynamically calculate করো
        calculated = float(monthly_charges * tenure)
        max_tc     = max(calculated + 1000.0, 10000.0)

        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=max_tc,
            value=calculated,
            step=1.0
        )

    st.markdown("---")

    # ── Predict Button ──
    if st.button("🔮 Predict Churn Probability"):

        input_data = {
            'gender'           : gender,
            'SeniorCitizen'    : 1 if senior_citizen == "Yes" else 0,
            'Partner'          : 1 if partner == "Yes" else 0,
            'Dependents'       : 1 if dependents == "Yes" else 0,
            'tenure'           : tenure,
            'PhoneService'     : 1 if phone_service == "Yes" else 0,
            'MultipleLines'    : 1 if multiple_lines == "Yes" else 0,
            'InternetService'  : internet_service,
            'OnlineSecurity'   : 1 if online_security == "Yes" else 0,
            'OnlineBackup'     : 1 if online_backup == "Yes" else 0,
            'DeviceProtection' : 1 if device_protection == "Yes" else 0,
            'TechSupport'      : 1 if tech_support == "Yes" else 0,
            'StreamingTV'      : 1 if streaming_tv == "Yes" else 0,
            'StreamingMovies'  : 1 if streaming_movies == "Yes" else 0,
            'Contract'         : contract,
            'PaperlessBilling' : 1 if paperless_billing == "Yes" else 0,
            'PaymentMethod'    : payment_method,
            'MonthlyCharges'   : monthly_charges,
            'TotalCharges'     : total_charges
        }

        with st.spinner("🔮 Predicting..."):
            input_df  = pd.DataFrame([input_data])
            input_eng = engineer_features(input_df)

            churn_prob = pipeline.predict_proba(input_eng)[0][1]
            churn_pred = pipeline.predict(input_eng)[0]

        # ── Results ──
        st.markdown("---")
        st.markdown("### 📊 Prediction Result")

        r1, r2, r3 = st.columns([2, 1, 2])

        with r1:
            fig = go.Figure(go.Indicator(
                mode  = "gauge+number+delta",
                value = round(churn_prob * 100, 1),
                title = {'text': "Churn Probability (%)"},
                delta = {'reference': 26.5},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar' : {
                        'color': "#e74c3c"
                        if churn_prob > 0.5 else "#2ecc71"
                    },
                    'steps': [
                        {'range': [0,  30], 'color': '#d5f5e3'},
                        {'range': [30, 60], 'color': '#fdebd0'},
                        {'range': [60,100], 'color': '#fadbd8'}
                    ],
                    'threshold': {
                        'line'     : {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value'    : 50
                    }
                }
            ))
            fig.update_layout(
                height=280,
                margin=dict(t=60, b=0, l=30, r=30)
            )
            st.plotly_chart(fig, use_container_width=True)

        with r2:
            st.markdown("<br><br><br>", unsafe_allow_html=True)
            if churn_pred == 1:
                st.markdown(
                    '<div class="churn-high">'
                    '⚠️ HIGH RISK<br><br>Will Churn'
                    '</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="churn-low">'
                    '✅ LOW RISK<br><br>Will Stay'
                    '</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<p style='text-align:center;"
                f"margin-top:10px; font-size:1.1rem'>"
                f"<b>{churn_prob*100:.1f}%</b> probability</p>",
                unsafe_allow_html=True
            )

        with r3:
            st.markdown("#### 💡 Recommendations")
            if churn_prob > 0.7:
                st.error("🚨 **Immediate action needed!**")
                st.markdown("""
                - 📞 Personal call from retention team
                - 🎁 Offer 3-month discount (30%)
                - ⬆️ Upgrade to annual contract
                - 🔧 Free tech support for 6 months
                """)
            elif churn_prob > 0.4:
                st.warning("⚠️ **Monitor closely**")
                st.markdown("""
                - 📧 Send loyalty reward email
                - 💡 Suggest better plan
                - 🎯 Targeted promotion offer
                - 📊 Monthly satisfaction check
                """)
            else:
                st.success("✅ **Customer is happy!**")
                st.markdown("""
                - 🌟 Upsell premium services
                - 🤝 Referral program invite
                - 🎖️ Loyalty points reward
                - 📱 Offer add-on services
                """)

        # ── Risk Factors ──
        st.markdown("---")
        st.markdown("### 🔍 Risk Factor Analysis")

        risk_data = []

        if contract == "Month-to-month":
            risk_data.append((
                "📋 Month-to-month contract",
                "HIGH", "#e74c3c", 3
            ))
        elif contract == "One year":
            risk_data.append((
                "📋 One year contract",
                "MEDIUM", "#f39c12", 2
            ))
        else:
            risk_data.append((
                "📋 Two year contract",
                "LOW", "#2ecc71", 1
            ))

        if tenure <= 6:
            risk_data.append((
                "🕐 New customer (≤6 months)",
                "HIGH", "#e74c3c", 3
            ))
        elif tenure <= 24:
            risk_data.append((
                "🕐 Mid-tenure customer",
                "MEDIUM", "#f39c12", 2
            ))
        else:
            risk_data.append((
                "🕐 Long-term customer (24+ months)",
                "LOW", "#2ecc71", 1
            ))

        if monthly_charges >= 70:
            risk_data.append((
                "💰 High monthly charges ($70+)",
                "HIGH", "#e74c3c", 3
            ))
        elif monthly_charges >= 50:
            risk_data.append((
                "💰 Medium monthly charges",
                "MEDIUM", "#f39c12", 2
            ))
        else:
            risk_data.append((
                "💰 Low monthly charges",
                "LOW", "#2ecc71", 1
            ))

        if internet_service == "Fiber optic":
            risk_data.append((
                "🌐 Fiber optic internet",
                "HIGH", "#e74c3c", 3
            ))

        if tech_support == "No":
            risk_data.append((
                "🔧 No tech support",
                "MEDIUM", "#f39c12", 2
            ))
        else:
            risk_data.append((
                "🔧 Has tech support",
                "LOW", "#2ecc71", 1
            ))

        if online_security == "No":
            risk_data.append((
                "🔒 No online security",
                "MEDIUM", "#f39c12", 2
            ))
        else:
            risk_data.append((
                "🔒 Has online security",
                "LOW", "#2ecc71", 1
            ))

        risk_data.sort(key=lambda x: x[3], reverse=True)

        rf1, rf2 = st.columns(2)
        for idx, (factor, level, color, _) in enumerate(risk_data):
            col = rf1 if idx % 2 == 0 else rf2
            with col:
                st.markdown(
                    f"<div style='padding:10px 15px;"
                    f"margin:6px 0;"
                    f"border-left:4px solid {color};"
                    f"background:#f8f9fa;"
                    f"border-radius:6px'>"
                    f"<b>{factor}</b><br>"
                    f"<span style='color:{color};"
                    f"font-size:0.85rem'>"
                    f"● Risk Level: <b>{level}</b>"
                    f"</span></div>",
                    unsafe_allow_html=True
                )

        # ── Input Summary ──
        st.markdown("---")
        with st.expander("📋 Input Summary (click to expand)"):
            summary_df = pd.DataFrame(
                list(input_data.items()),
                columns=['Feature', 'Value']
            )
            st.dataframe(summary_df, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("## 📊 Dataset Analytics")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Data load হয়নি!")
        st.stop()

    # ── Overview Metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total Customers", f"{len(df):,}")
    c2.metric(
        "🚨 Churned",
        f"{int(df['Churn'].sum()):,}",
        f"{df['Churn'].mean()*100:.1f}% rate"
    )
    c3.metric(
        "💰 Avg Monthly Charge",
        f"${df['MonthlyCharges'].mean():.2f}"
    )
    c4.metric(
        "📅 Avg Tenure",
        f"{df['tenure'].mean():.0f} months"
    )

    st.markdown("---")

    # ── Charts Row 1 ──
    ch1, ch2 = st.columns(2)

    with ch1:
        fig = px.histogram(
            df,
            x='tenure',
            color=df['Churn'].map({0: 'No Churn', 1: 'Churn'}),
            nbins=30,
            title='📅 Tenure Distribution by Churn',
            color_discrete_map={
                'No Churn': '#2ecc71',
                'Churn'   : '#e74c3c'
            },
            labels={'color': 'Status'}
        )
        fig.update_layout(bargap=0.05, legend_title='')
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig = px.box(
            df,
            x=df['Churn'].map({0: 'No Churn', 1: 'Churn'}),
            y='MonthlyCharges',
            color=df['Churn'].map({0: 'No Churn', 1: 'Churn'}),
            title='💰 Monthly Charges vs Churn',
            color_discrete_map={
                'No Churn': '#2ecc71',
                'Churn'   : '#e74c3c'
            }
        )
        fig.update_layout(
            xaxis_title='Churn Status',
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 2 ──
    ch3, ch4 = st.columns(2)

    with ch3:
        if 'Contract' in df.columns:
            contract_churn = (
                df.groupby('Contract')['Churn']
                .mean()
                .reset_index()
            )
            contract_churn.columns = ['Contract', 'Churn Rate']
            contract_churn['Label'] = (
                contract_churn['Churn Rate'] * 100
            ).round(1)

            fig = px.bar(
                contract_churn,
                x='Contract',
                y='Churn Rate',
                title='📋 Churn Rate by Contract Type',
                color='Churn Rate',
                color_continuous_scale='RdYlGn_r',
                text='Label'
            )
            fig.update_traces(
                texttemplate='%{text}%',
                textposition='outside'
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    with ch4:
        if 'InternetService' in df.columns:
            internet_churn = (
                df.groupby('InternetService')['Churn']
                .mean()
                .reset_index()
            )
            internet_churn.columns = ['Internet', 'Churn Rate']

            fig = px.pie(
                internet_churn,
                names='Internet',
                values='Churn Rate',
                title='🌐 Churn Rate by Internet Service',
                color_discrete_sequence=[
                    '#e74c3c', '#f39c12', '#2ecc71'
                ]
            )
            fig.update_traces(
                textposition='inside',
                textinfo='percent+label'
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Charts Row 3 ──
    ch5, ch6 = st.columns(2)

    with ch5:
        tenure_bins = pd.cut(
            df['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=['0-6', '7-12', '13-24', '25-48', '49-72']
        )
        tenure_churn = (
            df.groupby(tenure_bins, observed=True)['Churn']
            .mean()
            .reset_index()
        )
        tenure_churn.columns = ['Tenure Group', 'Churn Rate']
        tenure_churn['Label'] = (
            tenure_churn['Churn Rate'] * 100
        ).round(1)

        fig = px.bar(
            tenure_churn,
            x='Tenure Group',
            y='Churn Rate',
            title='🕐 Churn Rate by Tenure Group',
            color='Churn Rate',
            color_continuous_scale='RdYlGn_r',
            text='Label'
        )
        fig.update_traces(
            texttemplate='%{text}%',
            textposition='outside'
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with ch6:
        if 'TotalServices' in df.columns:
            svc_churn = (
                df.groupby('TotalServices')['Churn']
                .mean()
                .reset_index()
            )
            svc_churn.columns = ['Total Services', 'Churn Rate']

            fig = px.line(
                svc_churn,
                x='Total Services',
                y='Churn Rate',
                title='📱 Churn Rate by Number of Services',
                markers=True,
                line_shape='spline'
            )
            fig.update_traces(
                line_color='#e74c3c',
                marker_size=8
            )
            fig.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig, use_container_width=True)

    # ── Correlation Heatmap ──
    st.markdown("### 🔥 Feature Correlation Heatmap")
    num_cols = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'TotalServices', 'ContractRiskScore',
        'IsNewCustomer', 'IsHighValue', 'Churn'
    ]
    available = [c for c in num_cols if c in df.columns]
    corr = df[available].corr().round(3)

    fig = px.imshow(
        corr,
        text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
        aspect='auto'
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # ── Raw Data Preview ──
    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")

    col_f, col_r = st.columns([3, 1])
    with col_f:
        churn_filter = st.selectbox(
            "Filter by Churn:",
            ["All", "Churned (1)", "Not Churned (0)"]
        )
    with col_r:
        n_rows = st.number_input(
            "Rows to show", 5, 50, 10
        )

    if churn_filter == "Churned (1)":
        display_df = df[df['Churn'] == 1]
    elif churn_filter == "Not Churned (0)":
        display_df = df[df['Churn'] == 0]
    else:
        display_df = df

    st.dataframe(
        display_df.head(n_rows),
        use_container_width=True
    )
    st.caption(
        f"Showing {min(n_rows, len(display_df))} "
        f"of {len(display_df):,} rows"
    )