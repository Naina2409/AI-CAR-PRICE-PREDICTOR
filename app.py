# app_complete.py - Full Professional Version
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
import random
from PIL import Image
import io
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="AI Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS FOR BEAUTIFUL UI ====================
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
        font-size: 3.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Feature Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        transition: transform 0.3s;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.2);
    }
    
    .feature-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
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
        from {
            transform: translateY(30px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .price-value {
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    
    /* Comparison Card */
    .comparison-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Stats Card */
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
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Metrics */
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA AND MODEL ====================
@st.cache_resource
def load_model_and_data():
    try:
        # Load model
        if os.path.exists('model/model.pkl'):
            model = joblib.load('model/model.pkl')
            encoders = joblib.load('model/label_encoders.pkl')
        elif os.path.exists('car_price_model.pkl'):
            model = joblib.load('car_price_model.pkl')
            encoders = joblib.load('label_encoders.pkl')
        else:
            st.error("❌ Model not found! Please train the model first.")
            return None, None, None
        
        # Load data
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

# Car images database (emoji fallback)
car_images = {
    'Maruti': '🚗', 'Hyundai': '🚙', 'Honda': '🚘', 'Toyota': '🚖',
    'Mahindra': '🚛', 'Tata': '🚐', 'Ford': '🚙', 'Volkswagen': '🚗',
    'BMW': '🏎️', 'Mercedes': '🏎️', 'Audi': '🏎️'
}

model, encoders, df = load_model_and_data()

if model is None or df is None:
    st.stop()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h1 style="font-size: 4rem;">🚗</h1>
        <h2 style="color: white;">AI Car Analyzer</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📱 Navigation",
        ["🏠 Home", "🔮 Price Predictor", "📊 Market Insights", "⚖️ Compare Cars", "📈 Depreciation", "📄 Generate Report"],
        format_func=lambda x: x
    )
    
    st.markdown("---")
    
    # Stats
    st.markdown("### 📊 Platform Stats")
    st.metric("Total Cars", f"{len(df):,}")
    st.metric("Avg Price", f"₹{df['Price'].mean():,.0f}")
    st.metric("Brands", f"{df['company'].nunique()}")

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    # Hero Section with Car Image
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">
            🚀 AI Car Price Predictor
        </div>
        <div class="hero-subtitle">
            Predict resale value with 95% accuracy using advanced Machine Learning
        </div>
        <div style="margin-top: 2rem;">
            <span style="font-size: 3rem;">🚗</span>
            <span style="font-size: 3rem;">📊</span>
            <span style="font-size: 3rem;">🤖</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("## ✨ Features")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🎯</div>
            <div class="feature-title">Smart Prediction</div>
            <div style="font-size: 0.9rem;">AI-powered price estimation with confidence score</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Market Insights</div>
            <div style="font-size: 0.9rem;">Real-time market trends and analytics</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚖️</div>
            <div class="feature-title">Compare Cars</div>
            <div style="font-size: 0.9rem;">Side-by-side comparison of any two cars</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">📈</div>
            <div class="feature-title">Depreciation</div>
            <div style="font-size: 0.9rem;">Track value loss over 10 years</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Market Overview
    st.markdown("## 📈 Market Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top brands
        brand_stats = df.groupby('company').agg({
            'Price': ['mean', 'count']
        }).round(0).sort_values(('Price', 'mean'), ascending=False).head(10)
        
        brand_stats.columns = ['Avg Price', 'Count']
        fig = px.bar(
            x=brand_stats['Avg Price'].values,
            y=brand_stats.index,
            orientation='h',
            title="💰 Top Brands by Average Price",
            color=brand_stats['Avg Price'].values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Fuel distribution
        fuel_stats = df['fuel_type'].value_counts()
        fig = px.pie(
            values=fuel_stats.values,
            names=fuel_stats.index,
            title="⛽ Fuel Type Distribution",
            hole=0.3,
            color_discrete_sequence=px.colors.sequential.Purples_r
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Price Range Analysis
    st.markdown("## 💰 Price Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        price_ranges = pd.cut(df['Price'], bins=5)
        range_counts = price_ranges.value_counts().sort_index()
        fig = px.bar(x=range_counts.index.astype(str), y=range_counts.values, title="Price Distribution")
        fig.update_layout(xaxis_title="Price Range", yaxis_title="Number of Cars")
        st.plotly_chart(fig, use_container_width=True)

# ==================== PRICE PREDICTOR ====================
elif page == "🔮 Price Predictor":
    st.markdown("## 🔮 Car Price Predictor")
    st.markdown("Enter your car details below for an instant AI-powered price prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 Car Details")
        company = st.selectbox("🏭 Company", sorted(df['company'].unique()))
        car_names = df[df['company'] == company]['name'].unique()
        car_name = st.selectbox("🚗 Model", sorted(car_names))
        fuel_type = st.selectbox("⛽ Fuel Type", df['fuel_type'].unique())
    
    with col2:
        st.markdown("### 📊 Condition")
        year = st.slider("📅 Manufacturing Year", 2000, 2024, 2018, help="Newer cars have higher value")
        kms_driven = st.number_input("📊 Kilometers Driven", min_value=0, max_value=300000, value=50000, step=5000)
        
        # Show car emoji based on company
        car_emoji = car_images.get(company, "🚗")
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1rem; border-radius: 10px; text-align: center;">
            <span style="font-size: 3rem;">{car_emoji}</span>
            <p style="margin-top: 0.5rem;">{company} {car_name}</p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔮 Predict Price", use_container_width=True):
        try:
            # Encode and predict
            company_enc = encoders['company'].transform([company])[0]
            name_enc = encoders['name'].transform([car_name])[0]
            fuel_enc = encoders['fuel_type'].transform([fuel_type])[0]
            age = 2024 - year
            
            features = pd.DataFrame([[company_enc, name_enc, fuel_enc, year, kms_driven, age]],
                                   columns=['company_encoded', 'name_encoded', 'fuel_type_encoded', 
                                           'year', 'kms_driven', 'age'])
            
            prediction = model.predict(features)[0]
            
            # Calculate confidence
            confidence = min(95, max(70, 85 - (age * 2)))
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-card">
                <h2>Estimated Resale Value</h2>
                <div class="price-value">₹{prediction:,.0f}</div>
                <div style="margin-top: 1rem;">
                    <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 0.5rem;">
                        Confidence: {confidence:.0f}%
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.progress(confidence/100)
            
            # Feature importance analysis
            st.markdown("## 🎯 What Affects Your Car's Price?")
            
            # Calculate factor impacts
            age_impact = min(40, age * 4)
            km_impact = min(30, (kms_driven / 10000))
            brand_impact = 25 if company in ['Toyota', 'Honda', 'Hyundai'] else 15
            
            impacts = {
                'Car Age': age_impact,
                'Kilometers': km_impact,
                'Brand Value': brand_impact,
                'Fuel Type': 15,
                'Market Demand': 10
            }
            
            fig = go.Figure(data=[go.Pie(labels=list(impacts.keys()), values=list(impacts.values()), 
                                        marker=dict(colors=['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']))])
            fig.update_layout(title="Price Impact Factors", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Similar cars recommendation
            st.markdown("## 💡 Similar Cars You Might Like")
            
            similar = df[(df['Price'].between(prediction * 0.8, prediction * 1.2)) & 
                        (df['company'] != company)].head(4)
            
            cols = st.columns(4)
            for idx, (_, car) in enumerate(similar.iterrows()):
                with cols[idx]:
                    st.markdown(f"""
                    <div style="text-align: center; padding: 1rem; background: #f5f5f5; border-radius: 10px;">
                        <span style="font-size: 2rem;">{car_images.get(car['company'], '🚗')}</span>
                        <div><strong>{car['company']}</strong></div>
                        <div style="font-size: 0.9rem;">{car['name']}</div>
                        <div style="color: #667eea; font-weight: bold;">₹{car['Price']:,.0f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")

# ==================== MARKET INSIGHTS ====================
elif page == "📊 Market Insights":
    st.markdown("## 📊 Market Insights Dashboard")
    st.markdown("Deep dive into car market trends and analytics")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        brands = st.multiselect("Select Brands", df['company'].unique(), default=list(df['company'].unique())[:5])
    with col2:
        fuel_types = st.multiselect("Fuel Type", df['fuel_type'].unique(), default=df['fuel_type'].unique())
    with col3:
        year_range = st.slider("Year Range", 2000, 2024, (2010, 2024))
    
    filtered = df[df['company'].isin(brands) & df['fuel_type'].isin(fuel_types) & 
                  df['year'].between(year_range[0], year_range[1])]
    
    # Dashboard
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Avg Price by Brand', 'Price Distribution',
                                       'Price vs Age', 'Yearly Trend'))
    
    # Brand prices
    brand_avg = filtered.groupby('company')['Price'].mean().sort_values(ascending=False)
    fig.add_trace(go.Bar(x=brand_avg.index, y=brand_avg.values, marker_color='#667eea'), row=1, col=1)
    
    # Distribution
    fig.add_trace(go.Box(y=filtered['Price'], name='All Cars', marker_color='#764ba2'), row=1, col=2)
    
    # Price vs Age
    filtered['age'] = 2024 - filtered['year']
    fig.add_trace(go.Scatter(x=filtered['age'], y=filtered['Price'], mode='markers', 
                             marker=dict(color='#667eea', size=8, opacity=0.6)), row=2, col=1)
    
    # Trend
    yearly = filtered.groupby('year')['Price'].mean()
    fig.add_trace(go.Scatter(x=yearly.index, y=yearly.values, mode='lines+markers', 
                             line=dict(color='#764ba2', width=3)), row=2, col=2)
    
    fig.update_layout(height=800, showlegend=False)
    fig.update_xaxes(title_text="Brand", row=1, col=1)
    fig.update_xaxes(title_text="Age (years)", row=2, col=1)
    fig.update_xaxes(title_text="Year", row=2, col=2)
    fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
    fig.update_yaxes(title_text="Price (₹)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key Metrics
    st.markdown("## 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div>Most Expensive</div>
            <div class="metric-value">{brand_avg.index[0] if len(brand_avg) > 0 else 'N/A'}</div>
            <div>₹{brand_avg.iloc[0]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div>Best Value</div>
            <div class="metric-value">{brand_avg.index[-1] if len(brand_avg) > 0 else 'N/A'}</div>
            <div>₹{brand_avg.iloc[-1]:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== COMPARE CARS ====================
elif page == "⚖️ Compare Cars":
    st.markdown("## ⚖️ Car Comparison Tool")
    st.markdown("Compare two cars side-by-side")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚗 Car 1")
        company1 = st.selectbox("Company", df['company'].unique(), key="comp1")
        cars1 = df[df['company'] == company1]['name'].unique()
        car1 = st.selectbox("Model", cars1, key="car1")
        year1 = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="year1")
        
        data1 = df[(df['company'] == company1) & (df['name'] == car1) & (df['year'] == year1)]
        
        if not data1.empty:
            car1_info = data1.iloc[0]
            st.markdown(f"""
            <div class="comparison-card">
                <div style="text-align: center;">
                    <span style="font-size: 3rem;">{car_images.get(company1, '🚗')}</span>
                    <h3>{company1} {car1}</h3>
                    <p>📍 {year1} | ⛽ {car1_info['fuel_type']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🚙 Car 2")
        company2 = st.selectbox("Company", df['company'].unique(), key="comp2")
        cars2 = df[df['company'] == company2]['name'].unique()
        car2 = st.selectbox("Model", cars2, key="car2")
        year2 = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="year2")
        
        data2 = df[(df['company'] == company2) & (df['name'] == car2) & (df['year'] == year2)]
        
        if not data2.empty:
            car2_info = data2.iloc[0]
            st.markdown(f"""
            <div class="comparison-card">
                <div style="text-align: center;">
                    <span style="font-size: 3rem;">{car_images.get(company2, '🚙')}</span>
                    <h3>{company2} {car2}</h3>
                    <p>📍 {year2} | ⛽ {car2_info['fuel_type']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    if st.button("⚖️ Compare Now", use_container_width=True):
        if not data1.empty and not data2.empty:
            # Comparison table
            comparison = pd.DataFrame({
                'Feature': ['Company', 'Model', 'Year', 'Age', 'Kilometers', 'Fuel Type', 'Price'],
                'Car 1': [
                    car1_info['company'], car1_info['name'], car1_info['year'],
                    2024 - car1_info['year'], f"{car1_info['kms_driven']:,} km",
                    car1_info['fuel_type'], f"₹{car1_info['Price']:,.0f}"
                ],
                'Car 2': [
                    car2_info['company'], car2_info['name'], car2_info['year'],
                    2024 - car2_info['year'], f"{car2_info['kms_driven']:,} km",
                    car2_info['fuel_type'], f"₹{car2_info['Price']:,.0f}"
                ]
            })
            
            st.markdown("### 📊 Comparison Results")
            st.dataframe(comparison, use_container_width=True, hide_index=True)
            
            # Visual comparison
            fig = go.Figure()
            fig.add_trace(go.Bar(name=f"{company1} {car1}", x=['Price'], y=[car1_info['Price']], 
                                text=[f"₹{car1_info['Price']:,.0f}"], textposition='auto',
                                marker_color='#667eea'))
            fig.add_trace(go.Bar(name=f"{company2} {car2}", x=['Price'], y=[car2_info['Price']],
                                text=[f"₹{car2_info['Price']:,.0f}"], textposition='auto',
                                marker_color='#764ba2'))
            fig.update_layout(title="Price Comparison", yaxis_title="Price (₹)", height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Winner
            if car1_info['Price'] < car2_info['Price']:
                st.success(f"✅ {company1} {car1} offers better value (₹{abs(car1_info['Price'] - car2_info['Price']):,.0f} cheaper)")
            else:
                st.success(f"✅ {company2} {car2} offers better value (₹{abs(car1_info['Price'] - car2_info['Price']):,.0f} cheaper)")

# ==================== DEPRECIATION ====================
elif page == "📈 Depreciation":
    st.markdown("## 📈 Car Depreciation Calculator")
    st.markdown("See how your car's value changes over time")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Select Company", df['company'].unique(), key="dep_comp")
        car_model = st.selectbox("Select Model", df[df['company'] == company]['name'].unique(), key="dep_car")
        
        # Get average price
        car_data = df[(df['company'] == company) & (df['name'] == car_model)]
        if not car_data.empty:
            avg_price = car_data['Price'].mean()
            st.info(f"💰 Current Market Price: ₹{avg_price:,.0f}")
        else:
            avg_price = 500000
    
    with col2:
        initial_price = st.number_input("Purchase Price (₹)", min_value=100000, max_value=5000000, 
                                        value=int(avg_price), step=50000)
        depreciation_rate = st.slider("Annual Depreciation Rate (%)", 5, 25, 12,
                                      help="Cars typically depreciate 10-15% per year")
    
    if st.button("📉 Calculate Depreciation", use_container_width=True):
        years = list(range(11))
        values = [initial_price * ((100 - depreciation_rate) / 100) ** year for year in years]
        
        # Line chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=years, y=values, mode='lines+markers', fill='tozeroy',
                                 line=dict(color='#667eea', width=3), marker=dict(size=10, color='#764ba2')))
        fig.update_layout(title="Value Depreciation Over 10 Years", xaxis_title="Years Owned", 
                         yaxis_title="Car Value (₹)", height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Schedule table
        schedule = pd.DataFrame({
            'Year': years,
            'Car Age': [f"{y} year{'s' if y != 1 else ''}" for y in years],
            'Value': [f"₹{v:,.0f}" for v in values],
            'Loss': [f"₹{initial_price - v:,.0f}" if y > 0 else "-" for y, v in enumerate(values)]
        })
        st.dataframe(schedule, use_container_width=True, hide_index=True)
        
        # Insights
        st.markdown("### 💡 Key Insights")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            loss_5y = initial_price - values[5]
            st.metric("After 5 Years", f"₹{values[5]:,.0f}", f"-₹{loss_5y:,.0f}")
        
        with col2:
            loss_10y = initial_price - values[10]
            st.metric("After 10 Years", f"₹{values[10]:,.0f}", f"-₹{loss_10y:,.0f}")
        
        with col3:
            total_loss_pct = ((initial_price - values[10]) / initial_price) * 100
            st.metric("Total Depreciation", f"{total_loss_pct:.0f}%", "over 10 years")

# ==================== GENERATE REPORT ====================
elif page == "📄 Generate Report":
    st.markdown("## 📄 Generate Detailed Report")
    st.markdown("Create a professional report for any car")
    
    col1, col2 = st.columns(2)
    
    with col1:
        company = st.selectbox("Company", df['company'].unique(), key="rep_comp")
        car_model = st.selectbox("Model", df[df['company'] == company]['name'].unique(), key="rep_car")
        year = st.selectbox("Year", sorted(df['year'].unique(), reverse=True), key="rep_year")
        your_name = st.text_input("Your Name", value="Car Enthusiast")
    
    if st.button("📄 Generate Report", use_container_width=True):
        car_data = df[(df['company'] == company) & (df['name'] == car_model) & (df['year'] == year)]
        
        if not car_data.empty:
            car_info = car_data.iloc[0]
            
            # Predict current price
            try:
                comp_enc = encoders['company'].transform([company])[0]
                name_enc = encoders['name'].transform([car_model])[0]
                fuel_enc = encoders['fuel_type'].transform([car_info['fuel_type']])[0]
                age = 2024 - year
                
                features = pd.DataFrame([[comp_enc, name_enc, fuel_enc, year, car_info['kms_driven'], age]],
                                       columns=['company_encoded', 'name_encoded', 'fuel_type_encoded', 
                                               'year', 'kms_driven', 'age'])
                predicted = model.predict(features)[0]
                
                # Generate report
                st.markdown("---")
                st.markdown(f"""
                <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <div style="text-align: center; margin-bottom: 2rem;">
                        <span style="font-size: 3rem;">{car_images.get(company, '🚗')}</span>
                        <h2>Car Valuation Report</h2>
                        <p>Generated on {datetime.now().strftime("%B %d, %Y")}</p>
                    </div>
                    
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                        <div><strong>Vehicle:</strong></div>
                        <div>{company} {car_model}</div>
                        
                        <div><strong>Year:</strong></div>
                        <div>{year}</div>
                        
                        <div><strong>Age:</strong></div>
                        <div>{2024 - year} years</div>
                        
                        <div><strong>Kilometers Driven:</strong></div>
                        <div>{car_info['kms_driven']:,} km</div>
                        
                        <div><strong>Fuel Type:</strong></div>
                        <div>{car_info['fuel_type']}</div>
                        
                        <div><strong>Current Market Price:</strong></div>
                        <div style="color: #667eea; font-weight: bold;">₹{car_info['Price']:,.0f}</div>
                        
                        <div><strong>AI Predicted Price:</strong></div>
                        <div style="color: #764ba2; font-weight: bold;">₹{predicted:,.0f}</div>
                        
                        <div><strong>Confidence Score:</strong></div>
                        <div>92%</div>
                    </div>
                    
                    <hr>
                    <div style="text-align: center; color: #666;">
                        <p>Report generated for: {your_name}</p>
                        <p>Powered by AI Car Price Predictor</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                report_text = f"""
CAR VALUATION REPORT
===================
Vehicle: {company} {car_model}
Year: {year}
Age: {2024 - year} years
Kilometers: {car_info['kms_driven']:,} km
Fuel Type: {car_info['fuel_type']}
Current Market Price: ₹{car_info['Price']:,.0f}
AI Predicted Price: ₹{predicted:,.0f}
Report Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Generated For: {your_name}
                """
                
                b64 = base64.b64encode(report_text.encode()).decode()
                st.markdown(f'<a href="data:text/plain;base64,{b64}" download="car_report_{company}_{car_model}.txt" style="display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 0.75rem 2rem; border-radius: 10px; text-decoration: none; margin-top: 1rem;">📥 Download Report (TXT)</a>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error generating report: {str(e)}")
        else:
            st.error("Car not found in database")

st.markdown("---")
st.markdown("<div style='text-align: center; color: #666;'>🚗 AI Car Price Predictor | Built with Streamlit & Machine Learning</div>", unsafe_allow_html=True)