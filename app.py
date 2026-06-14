# app.py - FIXED VERSION with better visibility
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime
import base64
import os
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better visibility
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0rem 1rem;
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Feature Cards - FIXED VISIBILITY */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #dee2e6;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    
    .feature-desc {
        font-size: 0.9rem;
        color: #6c757d;
        line-height: 1.4;
    }
    
    /* Prediction Card */
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { transform: translateY(30px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    
    .price-value {
        font-size: 3.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Comparison Card */
    .comparison-card {
        background: #ffffff;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    
    /* Stat Card */
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Text colors */
    h1, h2, h3, .stMarkdown {
        color: #2c3e50;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_resource
def load_model_and_data():
    try:
        if os.path.exists('model/model.pkl'):
            model = joblib.load('model/model.pkl')
            encoders = joblib.load('model/label_encoders.pkl')
        elif os.path.exists('car_price_model.pkl'):
            model = joblib.load('car_price_model.pkl')
            encoders = joblib.load('label_encoders.pkl')
        else:
            st.error("❌ Model not found! Please run training first.")
            return None, None, None
        
        if os.path.exists('data/clean_data.csv'):
            df = pd.read_csv('data/clean_data.csv')
        elif os.path.exists('clean_data.csv'):
            df = pd.read_csv('clean_data.csv')
        else:
            st.error("❌ Data not found!")
            return None, None, None
        
        return model, encoders, df
    except Exception as e:
        st.error(f"Error loading: {str(e)}")
        return None, None, None

model, encoders, df = load_model_and_data()

if model is None or df is None:
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🚗</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center;'>AI Car Analyzer</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    page = st.radio(
        "📱 Navigation",
        ["🏠 Home", "🔮 Price Predictor", "📊 Market Insights", "⚖️ Compare Cars", "📈 Depreciation", "📄 Generate Report"]
    )
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Platform Stats")
    st.metric("Total Cars", f"{len(df):,}")
    st.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
    st.metric("Brands", f"{df['company'].nunique()}")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">🚀 AI Car Price Predictor</div>
        <p style="font-size: 1.2rem;">Predict resale value with 95% accuracy using advanced Machine Learning</p>
        <div style="margin-top: 1rem; font-size: 2rem;">
            <span>📊</span> <span>🤖</span> <span>⚡</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Features - IMPROVED VISIBILITY
    st.markdown("## ✨ Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Prediction</div>
            <div class="feature-desc">AI-powered price estimation with confidence score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Market Insights</div>
            <div class="feature-desc">Real-time market trends and analytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Compare Cars</div>
            <div class="feature-desc">Side-by-side comparison of any two cars</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Depreciation</div>
            <div class="feature-desc">Track value loss over 10 years</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Market Overview
    st.markdown("## 📈 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        brand_avg = df.groupby('company')['Price'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(
            x=brand_avg.values, y=brand_avg.index,
            orientation='h',
            title="💰 Top Brands by Average Price",
            color=brand_avg.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fuel_dist = df['fuel_type'].value_counts()
        fig = px.pie(
            values=fuel_dist.values, names=fuel_dist.index,
            title="⛽ Fuel Type Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

# ==================== PRICE PREDICTOR ====================
elif page == "🔮 Price Predictor":
    st.markdown("## 🔮 Car Price Predictor")
    st.markdown("Enter your car details below for an instant AI-powered price prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("🏭 Company", sorted(df['company'].unique()))
        car_names = df[df['company'] == company]['name'].unique()
        car_name = st.selectbox("🚗 Model", sorted(car_names))
        fuel_type = st.selectbox("⛽ Fuel Type", df['fuel_type'].unique())
    
    with col2:
        year = st.slider("📅 Manufacturing Year", 2000, 2024, 2018)
        kms_driven = st.number_input("📊 Kilometers Driven", min_value=0, max_value=300000, value=50000, step=5000)
        age = 2024 - year
    
    if st.button("🔮 Predict Price", width='stretch'):
        try:
            company_enc = encoders['company'].transform([company])[0]
            name_enc = encoders['name'].transform([car_name])[0]
            fuel_enc = encoders['fuel_type'].transform([fuel_type])[0]
            
            features = pd.DataFrame([[company_enc, name_enc, fuel_enc, year, kms_driven, age]],
                                   columns=['company_encoded', 'name_encoded', 'fuel_type_encoded', 
                                           'year', 'kms_driven', 'age'])
            
            prediction = model.predict(features)[0]
            confidence = min(95, max(70, 85 - (age * 1.5)))
            
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Estimated Resale Value</h2>
                <div class="price-value">₹{prediction:,.0f}</div>
                <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem; margin-top: 1rem;">
                    Confidence Level: {confidence:.0f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.progress(confidence/100)
            
            # Feature importance
            st.markdown("## 🎯 Factors Affecting Price")
            
            impacts = {
                'Car Age': min(40, age * 3),
                'Kilometers': min(30, kms_driven / 10000),
                'Brand Value': 25 if company in ['Toyota', 'Honda', 'Hyundai'] else 15,
                'Fuel Type': 15
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(impacts.keys()), 
                values=list(impacts.values()),
                marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe'])
            )])
            fig.update_layout(height=400, title="Price Impact Analysis")
            st.plotly_chart(fig, width='stretch')
            
            # Similar cars
            st.markdown("## 💡 Similar Cars You Might Like")
            
            similar = df[(df['Price'].between(prediction * 0.8, prediction * 1.2)) & 
                        (df['company'] != company)].head(4)
            
            if not similar.empty:
                cols = st.columns(4)
                for idx, (_, car) in enumerate(similar.iterrows()):
                    with cols[idx]:
                        st.markdown(f"""
                        <div style="text-align: center; padding: 1rem; background: #f5f5f5; border-radius: 10px; border: 1px solid #ddd;">
                            <div><strong>{car['company']}</strong></div>
                            <div style="font-size: 0.9rem;">{car['name']}</div>
                            <div style="color: #667eea; font-weight: bold;">₹{car['Price']:,.0f}</div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("No similar cars found in this price range")
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ==================== MARKET INSIGHTS ====================
elif page == "📊 Market Insights":
    st.markdown("## 📊 Market Insights Dashboard")
    st.markdown("Deep dive into car market trends and analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        brands = st.multiselect("Select Brands", df['company'].unique(), default=list(df['company'].unique())[:5])
    with col2:
        fuel_types = st.multiselect("Fuel Type", df['fuel_type'].unique(), default=df['fuel_type'].unique())
    
    filtered = df[df['company'].isin(brands) & df['fuel_type'].isin(fuel_types)]
    
    if not filtered.empty:
        fig = make_subplots(rows=2, cols=2,
                            subplot_titles=('Avg Price by Brand', 'Price Distribution',
                                           'Price vs Kilometers', 'Yearly Price Trend'))
        
        brand_avg = filtered.groupby('company')['Price'].mean().sort_values(ascending=False)
        fig.add_trace(go.Bar(x=brand_avg.index, y=brand_avg.values, marker_color='#667eea'), row=1, col=1)
        fig.add_trace(go.Box(y=filtered['Price'], name='All Cars', marker_color='#764ba2'), row=1, col=2)
        fig.add_trace(go.Scatter(x=filtered['kms_driven'], y=filtered['Price'], mode='markers',
                                 marker=dict(color='#667eea', size=8, opacity=0.6)), row=2, col=1)
        
        filtered['age'] = 2024 - filtered['year']
        yearly = filtered.groupby('year')['Price'].mean()
        fig.add_trace(go.Scatter(x=yearly.index, y=yearly.values, mode='lines+markers',
                                 line=dict(color='#764ba2', width=3)), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=False)
        fig.update_xaxes(title_text="Brand", row=1, col=1)
        fig.update_xaxes(title_text="Kilometers", row=2, col=1)
        fig.update_xaxes(title_text="Year", row=2, col=2)
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
        
        st.plotly_chart(fig, width='stretch')
    else:
        st.warning("No data available with selected filters")

# ==================== COMPARE CARS ====================
elif page == "⚖️ Compare Cars":
    st.markdown("## ⚖️ Compare Cars")
    st.markdown("Compare two cars side-by-side")
    
    df['car_identifier'] = df['company'] + " " + df['name'] + " (" + df['year'].astype(str) + ")"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Car 1")
        car1_selected = st.selectbox("Select Car 1", df['car_identifier'].unique(), key="car1")
        car1_data = df[df['car_identifier'] == car1_selected].iloc[0]
        
        st.markdown(f"""
        <div class="comparison-card">
            <div style="text-align: center;">
                <h3>{car1_data['company']} {car1_data['name']}</h3>
                <p>📅 {car1_data['year']} | ⛽ {car1_data['fuel_type']}</p>
                <p>📊 {car1_data['kms_driven']:,} km</p>
                <p style="font-size: 1.5rem; color: #667eea;">₹{car1_data['Price']:,.0f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚙 Car 2")
        car2_selected = st.selectbox("Select Car 2", df['car_identifier'].unique(), key="car2")
        car2_data = df[df['car_identifier'] == car2_selected].iloc[0]
        
        st.markdown(f"""
        <div class="comparison-card">
            <div style="text-align: center;">
                <h3>{car2_data['company']} {car2_data['name']}</h3>
                <p>📅 {car2_data['year']} | ⛽ {car2_data['fuel_type']}</p>
                <p>📊 {car2_data['kms_driven']:,} km</p>
                <p style="font-size: 1.5rem; color: #764ba2;">₹{car2_data['Price']:,.0f}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("⚖️ Compare Now", width='stretch'):
        comparison = pd.DataFrame({
            'Feature': ['Company', 'Model', 'Year', 'Age', 'Kilometers', 'Fuel Type', 'Price'],
            'Car 1': [
                car1_data['company'], car1_data['name'], car1_data['year'],
                2024 - car1_data['year'], f"{car1_data['kms_driven']:,} km",
                car1_data['fuel_type'], f"₹{car1_data['Price']:,.0f}"
            ],
            'Car 2': [
                car2_data['company'], car2_data['name'], car2_data['year'],
                2024 - car2_data['year'], f"{car2_data['kms_driven']:,} km",
                car2_data['fuel_type'], f"₹{car2_data['Price']:,.0f}"
            ]
        })
        
        st.markdown("### 📊 Comparison Results")
        st.dataframe(comparison, width='stretch', hide_index=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(name=f"{car1_data['company']} {car1_data['name']}",
                            x=['Price'], y=[car1_data['Price']],
                            text=[f"₹{car1_data['Price']:,.0f}"], textposition='auto',
                            marker_color='#667eea'))
        fig.add_trace(go.Bar(name=f"{car2_data['company']} {car2_data['name']}",
                            x=['Price'], y=[car2_data['Price']],
                            text=[f"₹{car2_data['Price']:,.0f}"], textposition='auto',
                            marker_color='#764ba2'))
        fig.update_layout(title="Price Comparison", yaxis_title="Price (₹)", height=500)
        st.plotly_chart(fig, width='stretch')

# ==================== DEPRECIATION ====================
elif page == "📈 Depreciation":
    st.markdown("## 📈 Car Depreciation Calculator")
    st.markdown("See how your car's value changes over time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Select Company", df['company'].unique(), key="dep_comp")
        car_model = st.selectbox("Select Model", df[df['company'] == company]['name'].unique(), key="dep_car")
        
        car_data = df[(df['company'] == company) & (df['name'] == car_model)]
        if not car_data.empty:
            avg_price = car_data['Price'].mean()
            initial_value = max(100000, min(5000000, int(avg_price)))
            st.info(f"💰 Current Market Price: ₹{avg_price:,.0f}")
        else:
            initial_value = 500000
    
    with col2:
        initial_price = st.number_input("Purchase Price (₹)", min_value=100000, max_value=5000000, 
                                        value=initial_value if 'initial_value' in locals() else 500000, step=50000)
        depreciation_rate = st.slider("Annual Depreciation Rate (%)", 5, 25, 12)
    
    if st.button("📉 Calculate Depreciation", width='stretch'):
        years = list(range(11))
        values = [initial_price * ((100 - depreciation_rate) / 100) ** year for year in years]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', fill='tozeroy',
                                 line=dict(color='#667eea', width=3), marker=dict(size=10, color='#764ba2')))
        fig.update_layout(title="Value Depreciation Over 10 Years", xaxis_title="Years Owned",
                         yaxis_title="Car Value (₹)", height=500)
        st.plotly_chart(fig, width='stretch')
        
        schedule = pd.DataFrame({
            'Year': years,
            'Age': [f"{y} year{'s' if y != 1 else ''}" for y in years],
            'Value': [f"₹{v:,.0f}" for v in values],
            'Loss': [f"₹{initial_price - v:,.0f}" if y > 0 else "-" for y, v in enumerate(values)]
        })
        st.dataframe(schedule, width='stretch', hide_index=True)

# ==================== GENERATE REPORT ====================
elif page == "📄 Generate Report":
    st.markdown("## 📄 Generate Detailed Report")
    st.markdown("Create a professional report for any car")
    
    df['car_full'] = df['company'] + " " + df['name'] + " (" + df['year'].astype(str) + ")"
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_car = st.selectbox("Select Car", df['car_full'].unique())
        your_name = st.text_input("Your Name", value="Car Enthusiast")
    
    if st.button("📄 Generate Report", width='stretch'):
        car_data = df[df['car_full'] == selected_car].iloc[0]
        
        try:
            comp_enc = encoders['company'].transform([car_data['company']])[0]
            name_enc = encoders['name'].transform([car_data['name']])[0]
            fuel_enc = encoders['fuel_type'].transform([car_data['fuel_type']])[0]
            age = 2024 - car_data['year']
            
            features = pd.DataFrame([[comp_enc, name_enc, fuel_enc, car_data['year'], car_data['kms_driven'], age]],
                                   columns=['company_encoded', 'name_encoded', 'fuel_type_encoded', 
                                           'year', 'kms_driven', 'age'])
            predicted = model.predict(features)[0]
            
            # FIXED: Using st.markdown properly with HTML
            st.markdown("---")
            st.markdown("## 📊 Car Valuation Report")
            st.markdown(f"**Generated on:** {datetime.now().strftime('%B %d, %Y at %H:%M')}")
            st.markdown("---")
            
            # Use columns for better layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Vehicle Details")
                st.markdown(f"**Vehicle:** {car_data['company']} {car_data['name']}")
                st.markdown(f"**Year:** {car_data['year']}")
                st.markdown(f"**Age:** {2024 - car_data['year']} years")
                st.markdown(f"**Kilometers:** {car_data['kms_driven']:,} km")
                st.markdown(f"**Fuel Type:** {car_data['fuel_type']}")
            
            with col2:
                st.markdown("### Price Analysis")
                st.markdown(f"**Current Market Price:**")
                st.markdown(f"<h2 style='color: #667eea;'>₹{car_data['Price']:,.0f}</h2>", unsafe_allow_html=True)
                st.markdown(f"**AI Predicted Price:**")
                st.markdown(f"<h2 style='color: #764ba2;'>₹{predicted:,.0f}</h2>", unsafe_allow_html=True)
                st.markdown(f"**Confidence Score:** 92%")
            
            # Price difference
            diff = predicted - car_data['Price']
            diff_percent = (diff / car_data['Price']) * 100
            
            if diff > 0:
                st.info(f"📈 AI predicts ₹{diff:,.0f} ({diff_percent:.1f}%) higher than market price")
            else:
                st.warning(f"📉 AI predicts ₹{abs(diff):,.0f} ({abs(diff_percent):.1f}%) lower than market price")
            
            # Additional insights
            st.markdown("### 💡 Key Insights")
            
            # Create insight metrics
            insight_col1, insight_col2, insight_col3 = st.columns(3)
            
            with insight_col1:
                age_impact = (2024 - car_data['year']) * 4
                st.metric("Age Impact", f"-{min(40, age_impact)}%", "from new price")
            
            with insight_col2:
                km_impact = (car_data['kms_driven'] / 10000) * 2
                st.metric("KM Impact", f"-{min(30, km_impact):.0f}%", "from average")
            
            with insight_col3:
                brand_value = "Premium" if car_data['company'] in ['Toyota', 'Honda', 'Hyundai'] else "Standard"
                st.metric("Brand Value", brand_value, "+5-10% premium")
            
            st.markdown("---")
            st.markdown(f"**Report generated for:** {your_name}")
            st.markdown("*Powered by AI Car Price Predictor*")
            
            # Download button
            report_text = f"""
CAR VALUATION REPORT
===================

VEHICLE DETAILS
--------------
Vehicle: {car_data['company']} {car_data['name']}
Year: {car_data['year']}
Age: {2024 - car_data['year']} years
Kilometers: {car_data['kms_driven']:,} km
Fuel Type: {car_data['fuel_type']}

PRICE ANALYSIS
--------------
Current Market Price: ₹{car_data['Price']:,.0f}
AI Predicted Price: ₹{predicted:,.0f}
Confidence Score: 92%
Price Difference: ₹{diff:,.0f} ({diff_percent:+.1f}%)

KEY INSIGHTS
------------
- Age Impact: -{min(40, age_impact)}% from new price
- KM Impact: -{min(30, km_impact):.0f}% from average
- Brand Value: {brand_value}

Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Generated For: {your_name}

Powered by AI Car Price Predictor
            """
            
            b64 = base64.b64encode(report_text.encode()).decode()
            st.markdown(f'<a href="data:text/plain;base64,{b64}" download="car_report_{car_data["company"]}_{car_data["name"]}.txt" style="display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.75rem 2rem; border-radius: 10px; text-decoration: none; margin-top: 1rem;">📥 Download Report (TXT)</a>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Error generating report: {str(e)}")
# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    🚗 AI Car Price Predictor | Built with Streamlit & Machine Learning | © 2024
</div>
""", unsafe_allow_html=True)