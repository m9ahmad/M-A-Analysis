import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. CONFIG ---
st.set_page_config(page_title="M&A Strategy AI", layout="wide", page_icon="📈")

# --- 2. STYLING ---
st.html("""
<style>
    div[data-testid="stMetric"] {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #4B5563;
    }
    .main { background-color: #0e1117; }
</style>
""")

# --- 3. DATA ENGINE (With Auto-Scaling Fix) ---
@st.cache_data
def get_data():
    try:
        df = pd.read_csv("acquisitions_update_2021.csv")
        df['Acquisition Price'] = pd.to_numeric(df['Acquisition Price'], errors='coerce')
        df['Acquisition Year'] = pd.to_numeric(df['Acquisition Year'], errors='coerce')
        
        # MISTAKE FIX: If prices are in full dollars (e.g. 1,000,000), 
        # convert them to Millions (1.0) so our 'Billions' math works.
        avg_val = df['Acquisition Price'].mean()
        if avg_val > 1000000:
            df['Acquisition Price'] = df['Acquisition Price'] / 1000000
            
        return df.dropna(subset=['Acquisition Year'])
    except:
        return pd.DataFrame()

df = get_data()

# --- 4. APP LOGIC ---
if df.empty:
    st.error("⚠️ DATASET NOT FOUND: Ensure 'acquisitions_update_2021.csv' is in your GitHub repo.")
else:
    # Sidebar
    st.sidebar.title("🛡️ IB Strategy Room")
    all_parents = sorted(df['Parent Company'].unique())
    selected_parents = st.sidebar.multiselect("Focus Acquirers", all_parents, default=['Google', 'Microsoft', 'Apple'])
    
    # Filtering
    f_df = df[df['Parent Company'].isin(selected_parents)]

    # Header
    st.title("🚀 M&A Strategic Value & Success Analytics")
    st.caption("Forensic analysis of technological consolidation and market premium trends.")

    # --- KPI SECTION (Mistake Corrected) ---
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.metric("Total Deals", len(f_df))
    
    with k2:
        # Now that data is in Millions, sum/1000 gives us real Billions
        total_bn = f_df['Acquisition Price'].sum() / 1000
        st.metric("Total Invested", f"${total_bn:,.2f}B")
    
    with k3:
        # Smart formatting for average deal size
        avg_m = f_df['Acquisition Price'].mean()
        if avg_m >= 1000:
            st.metric("Avg Deal Size", f"${avg_m/1000:,.2f}B")
        else:
            st.metric("Avg Deal Size", f"${avg_m:,.1f}M")
    
    with k4:
        top_sector = f_df['Business'].mode()[0] if not f_df.empty else "N/A"
        st.metric("Top Sector", top_sector)

    st.divider()

    # --- TABS ---
    tab1, tab2 = st.tabs(["📊 Market Dynamics", "🤖 Predictive Lab"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Accumulation Velocity")
            vel = f_df.groupby(['Acquisition Year', 'Parent Company']).size().reset_index(name='Deals')
            fig_line = px.line(vel, x='Acquisition Year', y='Deals', color='Parent Company', markers=True)
            st.plotly_chart(fig_line, use_container_width=True)
        
        with c2:
            st.subheader("Strategic Focus Areas")
            biz_focus = f_df['Business'].value_counts().head(10).reset_index()
            fig_bar = px.bar(biz_focus, x='count', y='Business', orientation='h', color='count')
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("🔍 Product Evolution Matrix")
        st.dataframe(
            f_df[['Acquired Company', 'Parent Company', 'Derived Products', 'Business']].dropna(subset=['Derived Products']).head(25),
            use_container_width=True
        )

    with tab2:
        st.subheader("🤖 Future Deal Price Estimator")
        p1, p2, p3 = st.columns(3)
        with p1:
            pred_parent = st.selectbox("Acquirer", all_parents)
        with p2:
            pred_biz = st.selectbox("Target Industry Segment", sorted(df['Business'].unique()))
        with p3:
            pred_year = st.number_input("Target Year", 2026, 2030, 2026)

        if st.button("Generate AI Valuation Estimate"):
            base_val = df[df['Parent Company'] == pred_parent]['Acquisition Price'].median()
            if np.isnan(base_val) or base_val == 0:
                base_val = df[df['Business'] == pred_biz]['Acquisition Price'].mean()
            
            prediction = base_val * (1 + (pred_year - 2021) * 0.05)
            st.success(f"Estimated Acquisition Price: **${prediction:,.2f} Million**")
