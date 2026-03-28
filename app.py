# app.py
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ─── Path Setup ────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src.feature_engineering import engineer_features

# ─── Page Config ───────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .churn-high {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1.5rem; border-radius: 15px;
        color: white; text-align: center;
        font-size: 1.3rem; font-weight: bold;
    }
    .churn-low {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem; border-radius: 15px;
        color: white; text-align: center;
        font-size: 1.3rem; font-weight: bold;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-size: 1rem;
        font-weight: 600; width: 100%;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─── Auto Train ────────────────────────────────────────
def auto_train():
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline as SKPipeline
    import xgboost as xgb
    from src.feature_engineering import clean_data, engineer_features

    data_path = os.path.join(
        ROOT_DIR, 'data', 'raw',
        'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )
    df = pd.read_csv(data_path)
    df = clean_data(df)
    df = engineer_features(df)

    processed_dir = os.path.join(ROOT_DIR, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    df.to_csv(
        os.path.join(processed_dir, 'cleaned_featured_data.csv'),
        index=False
    )

    numerical_features = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'AvgMonthlySpend', 'ChargeIncreaseRate',
        'TotalServices', 'ContractRiskScore'
    ]
    categorical_features = [
        'InternetService', 'Contract', 'PaymentMethod'
    ]
    binary_features = [
        'SeniorCitizen', 'Partner', 'Dependents',
        'PhoneService', 'MultipleLines', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
        'IsLongTermCustomer', 'IsNewCustomer', 'IsHighValue'
    ]

    all_features = (numerical_features +
                    categorical_features + binary_features)
    X = df[all_features]
    y = df['Churn']

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]),
         numerical_features),
        ('cat', Pipeline([('encoder', OneHotEncoder(
            drop='first', sparse_output=False,
            handle_unknown='ignore'
        ))]), categorical_features),
        ('bin', 'passthrough', binary_features)
    ])

    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    full_pipeline = SKPipeline([
        ('preprocessor', preprocessor),
        ('model', xgb.XGBClassifier(
            n_estimators=300, max_depth=5,
            learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3,
            scale_pos_weight=neg/pos,
            random_state=42, eval_metric='auc',
            verbosity=0
        ))
    ])
    full_pipeline.fit(X_train, y_train)

    models_dir = os.path.join(ROOT_DIR, 'models')
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, 'full_pipeline.pkl'), 'wb') as f:
        pickle.dump(full_pipeline, f)

    return full_pipeline


# ─── Load Model & Data ─────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(ROOT_DIR, 'models', 'full_pipeline.pkl')
    if not os.path.exists(model_path):
        st.info("⏳ First run — model training হচ্ছে... (2-3 মিনিট)")
        pipeline = auto_train()
        st.success("✅ Model ready!")
        return pipeline
    with open(model_path, 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    data_path = os.path.join(
        ROOT_DIR, 'data', 'processed',
        'cleaned_featured_data.csv'
    )
    if not os.path.exists(data_path):
        load_model()
    return pd.read_csv(data_path)

try:
    pipeline     = load_model()
    df           = load_data()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.error(f"⚠️ Error: {e}")


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
        "**Algorithm:** XGBoost\n\n"
        "**ROC-AUC:** ~0.85\n\n"
        "**F1 Score:** ~0.62\n\n"
        "**Dataset:** Telco (7,043 rows)"
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; font-size:0.85rem'>"
        "Built by <b>Mahabub</b></p>",
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════
if page == "🏠 Home":

    st.markdown("# 🔮 Customer Churn Predictor")
    st.markdown(
        "##### ML-powered system to predict which customers are likely to leave"
    )
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Dataset",    "7,043",   "customers")
    c2.metric("🎯 ROC-AUC",    "0.850",   "+vs baseline")
    c3.metric("⚡ Features",   "28",      "engineered")
    c4.metric("🏆 Best Model", "XGBoost", "tuned")

    st.markdown("---")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("### 🎯 Problem Statement")
        st.markdown("""
        Telecom কোম্পানিগুলোর জন্য customer churn
        একটা বড় সমস্যা।

        নতুন customer acquire করতে existing retain
        করার চেয়ে **5-7x বেশি খরচ** লাগে।

        এই system টা predict করে **কোন customer
        চলে যেতে পারে** — আগেই retention offer দেওয়া যায়।
        """)
    with col_r:
        st.markdown("### 🛠️ Technical Approach")
        st.markdown("""
        - **EDA:** 15+ visualizations
        - **Feature Eng.:** 7 নতুন business features
        - **Models:** LR, RF, XGBoost, LightGBM
        - **Tracking:** MLflow
        - **Tuning:** RandomizedSearchCV
        - **Deployment:** Hugging Face Spaces
        """)

    st.markdown("---")
    st.markdown("### 🔍 Key Business Insights")

    i1, i2, i3 = st.columns(3)
    with i1:
        st.error("📋 **Month-to-month**\n\nChurn: **42%**")
    with i2:
        st.warning("🌐 **Fiber optic**\n\nChurn: **42%**")
    with i3:
        st.warning("🕐 **New customers**\n\nChurn: **47%**")

    i4, i5, i6 = st.columns(3)
    with i4:
        st.success("📅 **Two-year contract**\n\nChurn: **3%**")
    with i5:
        st.warning("🔒 **No security**\n\nChurn: **41%**")
    with i6:
        st.success("👥 **Long-term**\n\nChurn: **~15%**")

    st.markdown("---")
    st.success("👈 Sidebar থেকে **Prediction** page এ যাও!")


# ══════════════════════════════════════════════════════
# PAGE 2 — PREDICTION
# ══════════════════════════════════════════════════════
elif page == "🔮 Prediction":

    st.markdown("## 🔮 Predict Customer Churn")
    st.markdown("Customer information দাও — model predict করবে।")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Model load হয়নি!")
        st.stop()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 👤 Customer Info")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner        = st.selectbox("Has Partner?", ["No", "Yes"])
        dependents     = st.selectbox("Has Dependents?", ["No", "Yes"])
        tenure         = st.slider("Tenure (months)", 0, 72, 12)

    with col2:
        st.markdown("#### 📱 Services")
        phone_service     = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines    = st.selectbox("Multiple Lines",
            ["No", "Yes", "No phone service"])
        internet_service  = st.selectbox("Internet Service",
            ["DSL", "Fiber optic", "No"])
        online_security   = st.selectbox("Online Security",
            ["No", "Yes", "No internet service"])
        online_backup     = st.selectbox("Online Backup",
            ["No", "Yes", "No internet service"])
        device_protection = st.selectbox("Device Protection",
            ["No", "Yes", "No internet service"])
        tech_support      = st.selectbox("Tech Support",
            ["No", "Yes", "No internet service"])
        streaming_tv      = st.selectbox("Streaming TV",
            ["No", "Yes", "No internet service"])
        streaming_movies  = st.selectbox("Streaming Movies",
            ["No", "Yes", "No internet service"])

    with col3:
        st.markdown("#### 💳 Billing")
        contract          = st.selectbox("Contract Type",
            ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing",
            ["Yes", "No"])
        payment_method    = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0, max_value=150.0,
            value=65.0, step=0.5
        )
        calculated  = float(monthly_charges * tenure)
        max_tc      = max(calculated + 1000.0, 10000.0)
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0, max_value=max_tc,
            value=calculated, step=1.0
        )

    st.markdown("---")

    if st.button("🔮 Predict Churn Probability"):
        input_data = {
            'gender'           : gender,
            'SeniorCitizen'    : 1 if senior_citizen=="Yes" else 0,
            'Partner'          : 1 if partner=="Yes" else 0,
            'Dependents'       : 1 if dependents=="Yes" else 0,
            'tenure'           : tenure,
            'PhoneService'     : 1 if phone_service=="Yes" else 0,
            'MultipleLines'    : 1 if multiple_lines=="Yes" else 0,
            'InternetService'  : internet_service,
            'OnlineSecurity'   : 1 if online_security=="Yes" else 0,
            'OnlineBackup'     : 1 if online_backup=="Yes" else 0,
            'DeviceProtection' : 1 if device_protection=="Yes" else 0,
            'TechSupport'      : 1 if tech_support=="Yes" else 0,
            'StreamingTV'      : 1 if streaming_tv=="Yes" else 0,
            'StreamingMovies'  : 1 if streaming_movies=="Yes" else 0,
            'Contract'         : contract,
            'PaperlessBilling' : 1 if paperless_billing=="Yes" else 0,
            'PaymentMethod'    : payment_method,
            'MonthlyCharges'   : monthly_charges,
            'TotalCharges'     : total_charges
        }

        with st.spinner("🔮 Predicting..."):
            input_df   = pd.DataFrame([input_data])
            input_eng  = engineer_features(input_df)
            churn_prob = pipeline.predict_proba(input_eng)[0][1]
            churn_pred = pipeline.predict(input_eng)[0]

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
                    'bar' : {'color': "#e74c3c"
                             if churn_prob > 0.5 else "#2ecc71"},
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
                    '⚠️ HIGH RISK<br><br>Will Churn</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<div class="churn-low">'
                    '✅ LOW RISK<br><br>Will Stay</div>',
                    unsafe_allow_html=True
                )
            st.markdown(
                f"<p style='text-align:center; margin-top:10px;"
                f"font-size:1.1rem'>"
                f"<b>{churn_prob*100:.1f}%</b> probability</p>",
                unsafe_allow_html=True
            )

        with r3:
            st.markdown("#### 💡 Recommendations")
            if churn_prob > 0.7:
                st.error("🚨 **Immediate action needed!**")
                st.markdown("""
                - 📞 Personal retention call
                - 🎁 30% discount offer
                - ⬆️ Annual contract upgrade
                - 🔧 Free tech support
                """)
            elif churn_prob > 0.4:
                st.warning("⚠️ **Monitor closely**")
                st.markdown("""
                - 📧 Loyalty reward email
                - 💡 Better plan suggestion
                - 🎯 Targeted promotion
                """)
            else:
                st.success("✅ **Customer is happy!**")
                st.markdown("""
                - 🌟 Upsell premium services
                - 🤝 Referral program
                - 🎖️ Loyalty rewards
                """)

        st.markdown("---")
        st.markdown("### 🔍 Risk Factor Analysis")

        risk_data = []
        if contract == "Month-to-month":
            risk_data.append(("📋 Month-to-month contract",
                              "HIGH", "#e74c3c", 3))
        elif contract == "One year":
            risk_data.append(("📋 One year contract",
                              "MEDIUM", "#f39c12", 2))
        else:
            risk_data.append(("📋 Two year contract",
                              "LOW", "#2ecc71", 1))

        if tenure <= 6:
            risk_data.append(("🕐 New customer (≤6 months)",
                              "HIGH", "#e74c3c", 3))
        elif tenure <= 24:
            risk_data.append(("🕐 Mid-tenure",
                              "MEDIUM", "#f39c12", 2))
        else:
            risk_data.append(("🕐 Long-term customer",
                              "LOW", "#2ecc71", 1))

        if monthly_charges >= 70:
            risk_data.append(("💰 High monthly charges",
                              "HIGH", "#e74c3c", 3))
        elif monthly_charges >= 50:
            risk_data.append(("💰 Medium charges",
                              "MEDIUM", "#f39c12", 2))
        else:
            risk_data.append(("💰 Low monthly charges",
                              "LOW", "#2ecc71", 1))

        if internet_service == "Fiber optic":
            risk_data.append(("🌐 Fiber optic internet",
                              "HIGH", "#e74c3c", 3))
        if tech_support == "No":
            risk_data.append(("🔧 No tech support",
                              "MEDIUM", "#f39c12", 2))
        else:
            risk_data.append(("🔧 Has tech support",
                              "LOW", "#2ecc71", 1))

        risk_data.sort(key=lambda x: x[3], reverse=True)

        rf1, rf2 = st.columns(2)
        for idx, (factor, level, color, _) in enumerate(risk_data):
            col = rf1 if idx % 2 == 0 else rf2
            with col:
                st.markdown(
                    f"<div style='padding:10px 15px; margin:6px 0;"
                    f"border-left:4px solid {color};"
                    f"background:#f8f9fa; border-radius:6px'>"
                    f"<b>{factor}</b><br>"
                    f"<span style='color:{color}; font-size:0.85rem'>"
                    f"● Risk: <b>{level}</b></span></div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")
        with st.expander("📋 Input Summary"):
            st.dataframe(
                pd.DataFrame(
                    list(input_data.items()),
                    columns=['Feature', 'Value']
                ),
                use_container_width=True
            )


# ══════════════════════════════════════════════════════
# PAGE 3 — ANALYTICS
# ══════════════════════════════════════════════════════
elif page == "📊 Analytics":

    st.markdown("## 📊 Dataset Analytics")
    st.markdown("---")

    if not MODEL_LOADED:
        st.error("Data load হয়নি!")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Total",   f"{len(df):,}")
    c2.metric("🚨 Churned",
              f"{int(df['Churn'].sum()):,}",
              f"{df['Churn'].mean()*100:.1f}%")
    c3.metric("💰 Avg Monthly",
              f"${df['MonthlyCharges'].mean():.2f}")
    c4.metric("📅 Avg Tenure",
              f"{df['tenure'].mean():.0f} mo")

    st.markdown("---")

    ch1, ch2 = st.columns(2)
    with ch1:
        fig = px.histogram(
            df, x='tenure',
            color=df['Churn'].map({0:'No Churn', 1:'Churn'}),
            nbins=30,
            title='📅 Tenure Distribution',
            color_discrete_map={
                'No Churn':'#2ecc71', 'Churn':'#e74c3c'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with ch2:
        fig = px.box(
            df,
            x=df['Churn'].map({0:'No Churn', 1:'Churn'}),
            y='MonthlyCharges',
            color=df['Churn'].map({0:'No Churn', 1:'Churn'}),
            title='💰 Monthly Charges vs Churn',
            color_discrete_map={
                'No Churn':'#2ecc71', 'Churn':'#e74c3c'
            }
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    ch3, ch4 = st.columns(2)
    with ch3:
        if 'Contract' in df.columns:
            cc = df.groupby('Contract')['Churn']\
                   .mean().reset_index()
            cc.columns = ['Contract', 'Churn Rate']
            cc['Label'] = (cc['Churn Rate']*100).round(1)
            fig = px.bar(
                cc, x='Contract', y='Churn Rate',
                title='📋 Churn by Contract',
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
            ic = df.groupby('InternetService')['Churn']\
                   .mean().reset_index()
            ic.columns = ['Internet', 'Churn Rate']
            fig = px.pie(
                ic, names='Internet', values='Churn Rate',
                title='🌐 Churn by Internet Service',
                color_discrete_sequence=[
                    '#e74c3c','#f39c12','#2ecc71'
                ]
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔥 Correlation Heatmap")
    num_cols  = ['tenure','MonthlyCharges','TotalCharges',
                 'TotalServices','ContractRiskScore',
                 'IsNewCustomer','IsHighValue','Churn']
    available = [c for c in num_cols if c in df.columns]
    corr      = df[available].corr().round(3)
    fig = px.imshow(
        corr, text_auto='.2f',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Matrix',
        aspect='auto'
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 📋 Dataset Preview")
    cf, cr = st.columns([3, 1])
    with cf:
        churn_filter = st.selectbox(
            "Filter:",
            ["All", "Churned (1)", "Not Churned (0)"]
        )
    with cr:
        n_rows = st.number_input("Rows", 5, 50, 10)

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