# Standard library imports
import calendar
import datetime
from datetime import timedelta
import io
import streamlit as st

# Data manipulation imports
import pandas as pd
import numpy as np

# Visualization imports
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ML imports
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt

# Custom color scheme
COLOR_PALETTE = {
    'primary': '#2E86C1',
    'secondary': '#28B463',
    'accent': '#E74C3C',
    'neutral': '#566573',
    'background': '#F8F9F9'
}

# Set page configuration
st.set_page_config(
    page_title="Pharmacy Analytics Dashboard",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
  
    div[data-testid="stMetricValue"] > div {
        font-size: 1.9rem !important;
    }
    div[data-testid="stMetricDelta"] > div {
        font-size: 1rem !important;
    }
            
    </style>
""", unsafe_allow_html=True)

# Display main header with custom styling
st.markdown("<h1 style='text-align: center; color: #2E86C1; padding: 20px;'>Pharmacy Analytics Dashboard</h1>", unsafe_allow_html=True)

# Load Data
@st.cache_data(ttl=3600)
def load_data():
    file_path = "pharmacy.xlsx"
    xls = pd.ExcelFile(file_path)
    
    # Load all sheets
    lists_df = pd.read_excel(xls, sheet_name="lists")
    daily_income_df = pd.read_excel(xls, sheet_name="Daily Income")
    inventory_purchases_df = pd.read_excel(xls, sheet_name="Inventory Purchases")
    expenses_df = pd.read_excel(xls, sheet_name="Expenses")
    
    # Clean up column names
    lists_df.columns = lists_df.columns.str.strip().str.lower()
    inventory_purchases_df.columns = inventory_purchases_df.columns.str.strip()
    expenses_df.columns = expenses_df.columns.str.strip()
    
    # Data preprocessing
    for df, date_col in [(daily_income_df, "Date"), 
                        (inventory_purchases_df, "Date"), 
                        (expenses_df, "Date")]:
        # Convert dates
        df["date"] = pd.to_datetime(df[date_col], errors='coerce')
        # Fill numeric columns with 0
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        df[numeric_columns] = df[numeric_columns].fillna(0)
    
    # Calculate derived columns for Daily Income
    daily_income_df["net_income"] = daily_income_df["Total"] - expenses_df["Expense Amount"] - inventory_purchases_df["Invoice Amount"]
    daily_income_df["deficit"] = daily_income_df["Total"] - daily_income_df["Gross Income_sys"]
    
    # Remove rows with null dates
    daily_income_df = daily_income_df.dropna(subset=["Date"])
    inventory_purchases_df = inventory_purchases_df.dropna(subset=["Date"])
    expenses_df = expenses_df.dropna(subset=["Date"])
    
    return {
        "lists": lists_df,
        "daily_income": daily_income_df,
        "inventory_purchases": inventory_purchases_df,
        "expenses": expenses_df
    }

# Load data
try:
    data = load_data()
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()


# ====================== SIDEBAR FILTERS ======================
with st.sidebar:
    st.markdown("### 🔍 Dashboard Filters")
    st.divider()
    
    # Date range filter
    st.markdown("#### 📅 Date Range")
    date_preset = st.selectbox(
        "Quick Select",
        ["Custom", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Year to Date", "All Time"],
        key="date_preset"
    )
    
    if date_preset == "Custom":
        date_range = st.date_input(
            "Select Date Range",
            value=(data["daily_income"]["date"].min().date(), 
                   data["daily_income"]["date"].max().date()),
            min_value=data["daily_income"]["date"].min().date(),
            max_value=data["daily_income"]["date"].max().date()
        )
        start_date, end_date = date_range if len(date_range) == 2 else (
            data["daily_income"]["date"].min().date(),
            data["daily_income"]["date"].max().date()
        )
    else:
        end_date = data["daily_income"]["date"].max().date()
        start_date = {
            "Last 7 Days": end_date - timedelta(days=7),
            "Last 30 Days": end_date - timedelta(days=30),
            "Last 90 Days": end_date - timedelta(days=90),
            "Year to Date": datetime.date(end_date.year, 1, 1),
            "All Time": data["daily_income"]["date"].min().date()
        }[date_preset]

    # Month filter
    st.markdown("#### 📅 Select Month")
    month_preset = st.selectbox(
        "Month",
        ["All", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
        key="month_preset"
    )
    
    if month_preset != "All":
        month_number = list(calendar.month_name).index(month_preset)
        filtered_data = {
            "daily_income": data["daily_income"][
                (data["daily_income"]["date"].dt.month == month_number) & 
                (data["daily_income"]["date"].dt.date >= start_date) & 
                (data["daily_income"]["date"].dt.date <= end_date)
            ].copy(),
            "inventory": data["inventory_purchases"][
                (data["inventory_purchases"]["date"].dt.month == month_number) & 
                (data["inventory_purchases"]["date"].dt.date >= start_date) & 
                (data["inventory_purchases"]["date"].dt.date <= end_date)
            ].copy(),
            "expenses": data["expenses"][
                (data["expenses"]["date"].dt.month == month_number) & 
                (data["expenses"]["date"].dt.date >= start_date) & 
                (data["expenses"]["date"].dt.date <= end_date)
            ].copy()
        }
    else:
        filtered_data = {
            "daily_income": data["daily_income"][
                (data["daily_income"]["date"].dt.date >= start_date) & 
                (data["daily_income"]["date"].dt.date <= end_date)
            ].copy(),
            "inventory": data["inventory_purchases"][
                (data["inventory_purchases"]["date"].dt.date >= start_date) & 
                (data["inventory_purchases"]["date"].dt.date <= end_date)
            ].copy(),
            "expenses": data["expenses"][
                (data["expenses"]["date"].dt.date >= start_date) & 
                (data["expenses"]["date"].dt.date <= end_date)
            ].copy()
        }
    # Additional filters
    st.markdown("#### 📦 Inventory")
    inventory_types = sorted(data["inventory_purchases"]["Inventory Type"].unique().tolist())
    selected_type = st.selectbox("Inventory Type", ["All"] + inventory_types, key="sidebar_inventory_type")
    
    st.markdown("#### 🏢 Companies")
    companies = sorted(data["inventory_purchases"]["Invoice Company"].unique().tolist())
    selected_company = st.selectbox("Company", ["All"] + companies, key="sidebar_company")
    
    st.markdown("#### 💰 Expenses")
    expense_types = sorted(data["expenses"]["Expense Type"].unique().tolist())
    selected_expense = st.selectbox("Expense Type", ["All"] + expense_types, key="sidebar_expense_type")
    
    # Apply additional filters
    if selected_type != "All":
        filtered_data["inventory"] = filtered_data["inventory"][
            filtered_data["inventory"]["Inventory Type"] == selected_type
        ]
    
    if selected_company != "All":
        filtered_data["inventory"] = filtered_data["inventory"][
            filtered_data["inventory"]["Invoice Company"] == selected_company
        ]
    
    if selected_expense != "All":
        filtered_data["expenses"] = filtered_data["expenses"][
            filtered_data["expenses"]["Expense Type"] == selected_expense
        ]

# ====================== MAIN DASHBOARD CONTENT ======================

# Main Dashboard Tabs
tab_overview, tab_revenue, tab_inventory, tab_expenses, tab_analytics, tab_ml, tab_search = st.tabs([
    "📊 Overview",
    "💰 Revenue",
    "📦 Inventory",
    "💸 Expenses",
    "📈 Analytics",
    "🤖 ML & Predictions",
    "🔍 Search & Reports"
])


# Overview Tab
with tab_overview:
    st.markdown("### 📊 Key Performance Indicators")

    # --- KPI Calculations (Robustness) ---
    if not filtered_data["daily_income"].empty:
        total_income = filtered_data["daily_income"]["Total"].sum()
        avg_daily_revenue = filtered_data["daily_income"]["Total"].mean()
    else:
        total_income = 0
        avg_daily_revenue = 0

    if not filtered_data["expenses"].empty:
        total_expenses = filtered_data["expenses"]["Expense Amount"].sum()
    else:
        total_expenses = 0

    if not filtered_data["inventory"].empty:
        total_purchases = filtered_data["inventory"]["Invoice Amount"].sum()
    else:
        total_purchases = 0

    net_profit = total_income - total_expenses - total_purchases
    money_deficit = filtered_data["daily_income"]["deficit"].sum() if not filtered_data["daily_income"].empty else 0

    # First Row - Main KPIs with robust calculations
    kpi_cols = st.columns(5)
    
    with kpi_cols[0]:
        st.metric("Total Revenue", f"EGP {total_income:,.2f}", 
                  delta=f"{(total_income/total_income*100 if total_income > 0 else 0):.1f}% of Total")
    with kpi_cols[1]:
        st.metric("Total Expenses", f"EGP {total_expenses:,.2f}",
                  delta=f"{(total_expenses/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")
    with kpi_cols[2]:
        st.metric("Total Purchases", f"EGP {total_purchases:,.2f}",
                  delta=f"{(total_purchases/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")
    with kpi_cols[3]:
        st.metric("Net Profit", f"EGP {net_profit:,.2f}",
                  delta=f"{(net_profit/total_income*100 if total_income > 0 else 0):.1f}% Margin")
    with kpi_cols[4]:
        st.metric("Money Deficit", f"EGP {money_deficit:,.2f}",
                  delta=f"{(money_deficit/total_income*100 if total_income > 0 else 0):.1f}% of Revenue")

    st.markdown("---")

# Second Row - Payment Methods
    st.markdown("### 💳 Payment Methods Analysis")
    payment_cols = st.columns(4)
    
    total_cash = filtered_data["daily_income"]["cash"].sum()
    total_visa = filtered_data["daily_income"]["visa"].sum()
    total_due = filtered_data["daily_income"]["due amount"].sum()
    
    with payment_cols[0]:
        st.metric("Cash Payments", f"EGP {total_cash:,.2f}",
                 delta=f"{(total_cash/total_income*100):.1f}% of Revenue")
    
    with payment_cols[1]:
        st.metric("Visa Payments", f"EGP {total_visa:,.2f}",
                 delta=f"{(total_visa/total_income*100):.1f}% of Revenue")
    
    with payment_cols[2]:
        st.metric("Due Amounts", f"EGP {total_due:,.2f}",
                 delta=f"{(total_due/total_income*100):.1f}% of Revenue")
    
    with payment_cols[3]:
        st.metric("System Income", f"EGP {filtered_data['daily_income']['Gross Income_sys'].sum():,.2f}",
                 delta=f"{(filtered_data['daily_income']['Gross Income_sys'].sum()/total_income*100):.1f}%")


    st.markdown("---")

    # --- Financial Health Indicators (New) ---
    st.markdown("### 📈 Financial Health Indicators")
    health_cols = st.columns(3)

    with health_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        fig_margin = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=profit_margin,
            title={"text": "Profit Margin (%)", "font": {"size": 20}},
            delta={'reference': 30, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "black"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 15], "color": "red"},
                    {"range": [15, 30], "color": "yellow"},
                    {"range": [30, 100], "color": "green"},
                ],
            }
        ))
        st.plotly_chart(fig_margin, use_container_width=True)

    with health_cols[1]:
        expense_ratio = (total_expenses / total_income * 100) if total_income > 0 else 0
        fig_expense_ratio = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=expense_ratio,
            title={"text": "Expense Ratio (%)", 'font': {'size': 20}},
            delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 25], 'color': "green"},
                    {'range': [25, 40], 'color': "yellow"},
                    {'range': [40, 100], 'color': "red"}
                ],
            }
        ))
        st.plotly_chart(fig_expense_ratio, use_container_width=True)

    with health_cols[2]:
        purchase_to_income_ratio = (total_purchases / total_income * 100) if total_income > 0 else 0
        fig_ratio = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=purchase_to_income_ratio,
            title={'text': "Purchases to Income Ratio (%)", 'font': {'size': 20}},
            delta={'reference': 60, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "black"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': 'yellow'},
                    {'range': [40, 60], 'color': 'green'},
                    {'range': [60, 100], 'color': 'red'}]
            }))
        st.plotly_chart(fig_ratio, use_container_width=True)

      # --- Critical KPI Notice (New) ---
    # Initialize notices for each category
    notices = {
        "profit": {"message": "", "color": ""},
        "expense": {"message": "", "color": ""}, 
        "inventory": {"message": "", "color": ""}
    }

    # Profit Margin Checks
    if profit_margin < 20:
        notices["profit"] = {
            "message": "Profit Margin is critically low! Focus on increasing revenue or reducing costs.",
            "color": "#EF5A6F"  # Light red
        }
    elif profit_margin < 50:
        notices["profit"] = {
            "message": "Profit Margin needs improvement. Consider strategies to increase profitability.",
            "color": "#FFB22C"  # Light orange
        }
    else:
        notices["profit"] = {
            "message": "Profit Margin is healthy.",
            "color": "#219C90"  # Light green
        }

    # Expense Ratio Checks
    if expense_ratio > 50:
        notices["expense"] = {
            "message": "Expense Ratio is very high! Immediate action is needed to control expenses.",
            "color": "#EF5A6F"
        }
    elif expense_ratio > 30:
        notices["expense"] = {
            "message": "Expense Ratio is above target. Review and optimize expenses.",
            "color": "#FFB22C"
        }
    else:
        notices["expense"] = {
            "message": "Expense Ratio is within acceptable range.",
            "color": "#219C90"
        }

    # Inventory to Income Ratpurchase_to_income_ratio
    if purchase_to_income_ratio > 60:
        notices["inventory"] = {
            "message": "Inventory purchases are too high compared to income! Review purchasing strategy.",
            "color": "#EF5A6F"
        }
    elif purchase_to_income_ratio > 40:
        notices["inventory"] = {
            "message": "Inventory to income ratio is healthy.",
            "color": "#219C90"
        }
    else:
        notices["inventory"] = {
            "message": "Inventory to income ratio is elevated. Consider optimizing purchases.",
            "color": "#FFB22C"
        }


    # Display notices in three columns
    cols = st.columns(3)
    for idx, (category, notice) in enumerate(notices.items()):
        with cols[idx]:
            st.markdown(
                f"""
                <div style="
                    border-radius: 10px;
                    text-align: center;
                    color: white;
                    font-size: 16px;
                    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
                    background-color: {notice['color']};
                    height: 100px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;">
                    <div style="
                        font-size: 18px;
                        margin-bottom: 8px;
                        text-transform: uppercase;
                        letter-spacing: 1px;
                        font-weight: bold;">
                        {category.title()}
                    </div>
                    <div>
                        {notice['message']}
                    </div>
                </div>
                """, 
                unsafe_allow_html=True
            )

    st.markdown("---")

    # --- Revenue vs. Expenses Chart (Enhanced) ---
    st.markdown("### 💰 Revenue vs. Expenses Analysis")
    if not filtered_data["daily_income"].empty and not filtered_data["expenses"].empty:
        daily_data = filtered_data["daily_income"].groupby("date")["Total"].sum().reset_index()
        daily_data = daily_data.merge(
            filtered_data["expenses"].groupby("date")["Expense Amount"].sum().reset_index(), 
            on="date", 
            how="left"
        )
        daily_data = daily_data.merge(
            filtered_data["inventory"].groupby("date")["Invoice Amount"].sum().reset_index(), 
            on="date", 
            how="left"
        )
        daily_data["Expense Amount"] = daily_data["Expense Amount"].fillna(0)
        daily_data["Invoice Amount"] = daily_data["Invoice Amount"].fillna(0)
        daily_data["net_profit"] = daily_data["Total"] - daily_data["Expense Amount"] - daily_data["Invoice Amount"]

        fig_rev_exp = go.Figure()
        # Add traces
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["date"], 
            y=daily_data["Total"], 
            name="Revenue",
            line=dict(color=COLOR_PALETTE["primary"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["date"], 
            y=daily_data["Expense Amount"], 
            name="Expenses",
            line=dict(color=COLOR_PALETTE["accent"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["date"], 
            y=daily_data["Invoice Amount"], 
            name="Purchases",
            line=dict(color=COLOR_PALETTE["neutral"]),
            fill='tozeroy'
        ))
        fig_rev_exp.add_trace(go.Scatter(
            x=daily_data["date"], 
            y=daily_data["net_profit"], 
            name="Net Profit",
            line=dict(color=COLOR_PALETTE["secondary"], dash='dash')
        ))

        fig_rev_exp.update_layout(
            title="Daily Revenue, Expenses, Purchases, and Profit Analysis",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig_rev_exp, use_container_width=True)
    else:
        st.warning("Insufficient data to display Revenue vs Expenses analysis.")

    st.markdown("---")

    # Third Row - Daily Trends
    st.markdown("### 📈 Daily Performance")
    daily_cols = st.columns(2)
    
    with daily_cols[0]:
        # Daily Revenue vs System Revenue
        daily_comparison = filtered_data["daily_income"].groupby("Date").agg({
            "Total": "sum",
            "Gross Income_sys": "sum"
        }).reset_index()
        
        fig_daily_comp = go.Figure()
        fig_daily_comp.add_trace(go.Scatter(
            x=daily_comparison["Date"],
            y=daily_comparison["Total"],
            name="Actual Revenue",
            line=dict(color=COLOR_PALETTE["primary"])
        ))
        fig_daily_comp.add_trace(go.Scatter(
            x=daily_comparison["Date"],
            y=daily_comparison["Gross Income_sys"],
            name="System Revenue",
            line=dict(color=COLOR_PALETTE["secondary"])
        ))
        fig_daily_comp.update_layout(
            title="Daily Revenue vs System Revenue",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_daily_comp, use_container_width=True)
    
    with daily_cols[1]:
        # Daily Payment Methods
        daily_payments = filtered_data["daily_income"].groupby("Date").agg({
            "cash": "sum",
            "visa": "sum",
            "due amount": "sum"
        }).reset_index()
        
        fig_payments = go.Figure()
        for payment_type in ["cash", "visa", "due amount"]:
            fig_payments.add_trace(go.Bar(
                x=daily_payments["Date"],
                y=daily_payments[payment_type],
                name=payment_type.title()
            ))
        fig_payments.update_layout(
            title="Daily Payment Methods Distribution",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            barmode="stack"
        )
        st.plotly_chart(fig_payments, use_container_width=True)

    st.markdown("---")

    # Fourth Row - Expense Analysis
    st.markdown("### 💸 Expense Breakdown")
    expense_cols = st.columns(2)
    
    with expense_cols[0]:
        # Expense Types Distribution
        expense_by_type = filtered_data["expenses"].groupby("Expense Type")["Expense Amount"].sum()
        fig_expense = go.Figure(data=[go.Pie(
            labels=expense_by_type.index,
            values=expense_by_type.values,
            hole=0.4
        )])
        fig_expense.update_layout(
            title="Expense Distribution by Type",
            template="plotly_white"
        )
        st.plotly_chart(fig_expense, use_container_width=True)
    
    with expense_cols[1]:
        # Daily Expenses Trend
        daily_expenses = filtered_data["expenses"].groupby("Date")["Expense Amount"].sum().reset_index()
        fig_exp_trend = go.Figure()
        fig_exp_trend.add_trace(go.Scatter(
            x=daily_expenses["Date"],
            y=daily_expenses["Expense Amount"],
            mode="lines+markers",
            line=dict(color=COLOR_PALETTE["accent"])
        ))
        fig_exp_trend.update_layout(
            title="Daily Expenses Trend",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_exp_trend, use_container_width=True)

    st.markdown("---")

    # Fifth Row - Inventory Analysis
    st.markdown("### 📦 Inventory Insights")
    inventory_cols = st.columns(2)
    
    with inventory_cols[0]:
        # Inventory by Company
        inv_by_company = filtered_data["inventory"].groupby("Invoice Company")["Invoice Amount"].sum().sort_values(ascending=True)
        fig_inv_company = go.Figure(data=[go.Bar(
            y=inv_by_company.index,
            x=inv_by_company.values,
            orientation="h",
            marker_color=COLOR_PALETTE["primary"]
        )])
        fig_inv_company.update_layout(
            title="Purchases by Company",
            xaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_inv_company, use_container_width=True)
    
    with inventory_cols[1]:
        # Inventory Types
        inv_by_type = filtered_data["inventory"].groupby("Inventory Type")["Invoice Amount"].sum()
        fig_inv_type = go.Figure(data=[go.Pie(
            labels=inv_by_type.index,
            values=inv_by_type.values,
            hole=0.4
        )])
        fig_inv_type.update_layout(
            title="Distribution by Inventory Type",
            template="plotly_white"
        )
        st.plotly_chart(fig_inv_type, use_container_width=True)

    st.markdown("#### 📥 Export Complete Dashboard Report")
    if st.button("Generate Complete Dashboard Report"):
        with st.spinner("Generating comprehensive report..."):
            # Create Excel writer
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Financial Overview
                financial_summary = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Total Expenses', 'Total Purchases', 'Net Profit', 'Money Deficit'],
                    'Amount': [total_income, total_expenses, total_purchases, net_profit, money_deficit],
                    'Percentage of Revenue': [100, 
                                           (total_expenses/total_income*100) if total_income > 0 else 0,
                                           (total_purchases/total_income*100) if total_income > 0 else 0,
                                           (net_profit/total_income*100) if total_income > 0 else 0,
                                           (money_deficit/total_income*100) if total_income > 0 else 0]
                })
                financial_summary.to_excel(writer, sheet_name='Financial Overview', index=False)
                total_revenue = filtered_data["daily_income"]["Total"].sum()
                cash_percentage = (filtered_data["daily_income"]["cash"].sum() / total_revenue * 100) if total_revenue > 0 else 0
                visa_percentage = (filtered_data["daily_income"]["visa"].sum() / total_revenue * 100) if total_revenue > 0 else 0
                due_percentage = (filtered_data["daily_income"]["due amount"].sum() / total_revenue * 100) if total_revenue > 0 else 0

                # Payment Methods Analysis
                payment_summary = pd.DataFrame({
                    'Payment Type': ['Cash', 'Visa', 'Due Amount'],
                    'Amount': [total_cash, total_visa, total_due],
                    'Percentage': [cash_percentage, visa_percentage, due_percentage]
                })
                payment_summary.to_excel(writer, sheet_name='Payment Analysis', index=False)
                
                # Daily Performance
                daily_performance = filtered_data["daily_income"].groupby("date").agg({
                    'Total': 'sum',
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum',
                    'Gross Income_sys': 'sum',
                    'deficit': 'sum'
                }).round(2)
                daily_performance.to_excel(writer, sheet_name='Daily Performance')
                
                # Expense Analysis
                expense_analysis = filtered_data["expenses"].pivot_table(
                    values='Expense Amount',
                    index='date',
                    columns='Expense Type',
                    aggfunc='sum',
                    fill_value=0
                ).round(2)
                expense_analysis.to_excel(writer, sheet_name='Expense Analysis')
                
                # Inventory Analysis
                inventory_analysis = filtered_data["inventory"].pivot_table(
                    values='Invoice Amount',
                    index='date',
                    columns=['Inventory Type', 'Invoice Company'],
                    aggfunc='sum',
                    fill_value=0
                ).round(2)
                inventory_analysis.to_excel(writer, sheet_name='Inventory Analysis')
                
                # KPI Metrics
                kpi_metrics = pd.DataFrame({
                    'Metric': ['Profit Margin', 'Expense Ratio', 'Purchase to Income Ratio'],
                    'Value': [profit_margin, expense_ratio, purchase_to_income_ratio],
                    'Status': [
                        'Healthy' if profit_margin >= 30 else 'Needs Improvement' if profit_margin >= 15 else 'Critical',
                        'Good' if expense_ratio <= 25 else 'Warning' if expense_ratio <= 40 else 'Critical',
                        'Optimal' if purchase_to_income_ratio <= 40 else 'Warning' if purchase_to_income_ratio <= 60 else 'Critical'
                    ]
                })
                kpi_metrics.to_excel(writer, sheet_name='KPI Metrics', index=False)
                
                # Create a worksheet for charts
                workbook = writer.book
                worksheet = workbook.add_worksheet('Charts')
                
                # Add title formats
                title_format = workbook.add_format({
                    'bold': True,
                    'font_size': 14,
                    'align': 'center',
                    'valign': 'vcenter'
                })
                
                # Add charts
                revenue_chart = workbook.add_chart({'type': 'line'})
                revenue_chart.add_series({
                    'name': 'Revenue',
                    'categories': '=Daily Performance!$A$2:$A$' + str(len(daily_performance) + 1),
                    'values': '=Daily Performance!$B$2:$B$' + str(len(daily_performance) + 1),
                })
                revenue_chart.set_title({'name': 'Revenue Trend'})
                worksheet.insert_chart('A1', revenue_chart)

            # Create download button
            output.seek(0)
            st.download_button(
                label="📥 Download Complete Report",
                data=output,
                file_name=f"pharmacy_complete_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Revenue Tab
with tab_revenue:
    st.markdown("### 💰 Revenue Analysis")
    
    # Main Revenue Metrics
    st.markdown("#### 📊 Primary Revenue Metrics")
    revenue_kpi_cols = st.columns(4)
    
    with revenue_kpi_cols[0]:
        total_revenue = filtered_data["daily_income"]["Total"].sum()
        st.metric("Total Revenue", f"EGP {total_revenue:,.2f}")
        st.markdown(f"**Daily Average:** EGP {filtered_data['daily_income']['Total'].mean():,.2f}")
        st.markdown(f"**Monthly Average:** EGP {total_revenue/((filtered_data['daily_income']['date'].max() - filtered_data['daily_income']['date'].min()).days/30):,.2f}")
    
    with revenue_kpi_cols[1]:
        cash_percentage = (filtered_data["daily_income"]["cash"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.metric("Cash Revenue %", f"{cash_percentage:.1f}%")
        st.markdown(f"**Cash Total:** EGP {filtered_data['daily_income']['cash'].sum():,.2f}")
        st.markdown(f"**Daily Cash Avg:** EGP {filtered_data['daily_income']['cash'].mean():,.2f}")
    
    with revenue_kpi_cols[2]:
        visa_percentage = (filtered_data["daily_income"]["visa"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.metric("Visa Revenue %", f"{visa_percentage:.1f}%")
        st.markdown(f"**Visa Total:** EGP {filtered_data['daily_income']['visa'].sum():,.2f}")
        st.markdown(f"**Daily Visa Avg:** EGP {filtered_data['daily_income']['visa'].mean():,.2f}")
    
    with revenue_kpi_cols[3]:
        due_percentage = (filtered_data["daily_income"]["due amount"].sum() / total_revenue * 100) if total_revenue > 0 else 0
        st.metric("Due Amount %", f"{due_percentage:.1f}%")
        st.markdown(f"**Due Total:** EGP {filtered_data['daily_income']['due amount'].sum():,.2f}")
        st.markdown(f"**Daily Due Avg:** EGP {filtered_data['daily_income']['due amount'].mean():,.2f}")

    # Revenue Growth Analysis
    st.markdown("#### 📈 Revenue Growth Analysis")
    growth_cols = st.columns(2)
    
    with growth_cols[0]:
        # Month-over-Month Growth
        monthly_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.strftime('%Y-%m'))["Total"].sum()
        mom_growth = ((monthly_revenue.iloc[-1] - monthly_revenue.iloc[-2]) / monthly_revenue.iloc[-2] * 100) if len(monthly_revenue) >= 2 else 0
        st.metric("Month-over-Month Growth", f"{mom_growth:.1f}%")
        
        # Weekly Growth Trend
        weekly_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.strftime('%Y-%W'))["Total"].sum()
        fig_weekly = go.Figure()
        fig_weekly.add_trace(go.Scatter(x=weekly_revenue.index, y=weekly_revenue.values, mode='lines+markers'))
        fig_weekly.update_layout(title="Weekly Revenue Trend", height=300)
        st.plotly_chart(fig_weekly, use_container_width=True)
    
    with growth_cols[1]:
        # Revenue Distribution by Day of Week
        dow_revenue = filtered_data["daily_income"].groupby(filtered_data["daily_income"]["date"].dt.day_name())["Total"].agg(["mean", "std"])
        fig_dow = go.Figure()
        fig_dow.add_trace(go.Bar(x=dow_revenue.index, y=dow_revenue["mean"], error_y=dict(type='data', array=dow_revenue["std"])))
        fig_dow.update_layout(title="Average Revenue by Day of Week", height=300)
        st.plotly_chart(fig_dow, use_container_width=True)

    # Payment Analysis
    st.markdown("#### 💳 Detailed Payment Analysis")
    payment_cols = st.columns(3)
    
    with payment_cols[0]:
        # Payment Method Trends
        payment_trends = filtered_data["daily_income"][["date", "cash", "visa", "due amount"]].melt(id_vars=["date"])
        fig_payment_trends = px.line(payment_trends, x="date", y="value", color="variable", title="Payment Method Trends")
        fig_payment_trends.update_layout(height=300)
        st.plotly_chart(fig_payment_trends, use_container_width=True)
        
    with payment_cols[1]:
        # Daily Payment Mix
        daily_mix = filtered_data["daily_income"][["cash", "visa", "due amount"]].div(filtered_data["daily_income"]["Total"], axis=0)
        fig_mix = go.Figure()
        for col in daily_mix.columns:
            fig_mix.add_trace(go.Box(y=daily_mix[col], name=col))
        fig_mix.update_layout(title="Daily Payment Mix Distribution", height=300)
        st.plotly_chart(fig_mix, use_container_width=True)
        
    with payment_cols[2]:
        # Payment Method Correlations
        payment_corr = filtered_data["daily_income"][["cash", "visa", "due amount"]].corr()
        fig_corr = go.Figure(data=go.Heatmap(z=payment_corr, x=payment_corr.columns, y=payment_corr.index))
        fig_corr.update_layout(title="Payment Method Correlations", height=300)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Revenue Performance Indicators
    st.markdown("#### 🎯 Revenue Performance Indicators")
    perf_cols = st.columns(4)
    
    with perf_cols[0]:
        revenue_volatility = filtered_data["daily_income"]["Total"].std() / filtered_data["daily_income"]["Total"].mean()
        st.metric("Revenue Volatility", f"{revenue_volatility:.2f}")
        
    with perf_cols[1]:
        revenue_skewness = filtered_data["daily_income"]["Total"].skew()
        st.metric("Revenue Skewness", f"{revenue_skewness:.2f}")
        
    with perf_cols[2]:
        peak_revenue = filtered_data["daily_income"]["Total"].max()
        st.metric("Peak Revenue", f"EGP {peak_revenue:,.2f}")
        
    with perf_cols[3]:
        revenue_consistency = (filtered_data["daily_income"]["Total"] > filtered_data["daily_income"]["Total"].mean()).mean() * 100
        st.metric("Above Average Days", f"{revenue_consistency:.1f}%")

    # Revenue Forecasting
    st.markdown("#### 🔮 Revenue Forecasting")
    forecast_cols = st.columns([2, 1])
    
    with forecast_cols[0]:
        # Simple Moving Averages
        ma_periods = [7, 14, 30]
        revenue_df = filtered_data["daily_income"][["date", "Total"]].set_index("date")
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=revenue_df.index, y=revenue_df["Total"], name="Actual"))
        
        for period in ma_periods:
            ma = revenue_df["Total"].rolling(period).mean()
            fig_ma.add_trace(go.Scatter(x=revenue_df.index, y=ma, name=f"{period}-day MA"))
            
        fig_ma.update_layout(title="Revenue Moving Averages", height=400)
        st.plotly_chart(fig_ma, use_container_width=True)
        
    with forecast_cols[1]:
        # Revenue Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(x=filtered_data["daily_income"]["Total"], nbinsx=30))
        fig_dist.update_layout(title="Revenue Distribution", height=400)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Revenue Segments Analysis
    st.markdown("#### 📊 Revenue Segments")
    
    # Create revenue segments
    filtered_data["daily_income"]["revenue_segment"] = pd.qcut(filtered_data["daily_income"]["Total"], 
                                                             q=4, 
                                                             labels=["Low", "Medium-Low", "Medium-High", "High"])
    
    segment_stats = filtered_data["daily_income"].groupby("revenue_segment").agg({
        "Total": ["count", "mean", "sum"],
        "cash": "sum",
        "visa": "sum",
        "due amount": "sum"
    }).round(2)
    
    st.dataframe(segment_stats, use_container_width=True)

    st.markdown("#### 📊 Revenue Reports")
    report_cols = st.columns(3)
    
    with report_cols[0]:
        if st.button("Export Detailed Revenue Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Daily Revenue Details
                daily_revenue = filtered_data["daily_income"].groupby("date").agg({
                    'Total': 'sum',
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum',
                    'Gross Income_sys': 'sum',
                    'net_income': 'sum',
                    'deficit': 'sum'
                }).round(2)
                daily_revenue.to_excel(writer, sheet_name='Daily Revenue')
                
                # Payment Method Analysis
                payment_analysis = pd.DataFrame({
                    'Method': ['Cash', 'Visa', 'Due Amount'],
                    'Total Amount': [total_cash, total_visa, total_due],
                    'Percentage': [cash_percentage, visa_percentage, due_percentage],
                    'Daily Average': [
                        filtered_data['daily_income']['cash'].mean(),
                        filtered_data['daily_income']['visa'].mean(),
                        filtered_data['daily_income']['due amount'].mean()
                    ]
                })
                payment_analysis.to_excel(writer, sheet_name='Payment Analysis', index=False)
                
                # Revenue Growth
                monthly_growth = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.strftime('%Y-%m')
                ).agg({
                    'Total': ['sum', 'mean', 'std'],
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum'
                }).round(2)
                monthly_growth.to_excel(writer, sheet_name='Monthly Analysis')
                
                # Revenue Segments
                segment_analysis = segment_stats.copy()
                segment_analysis.to_excel(writer, sheet_name='Revenue Segments')
                
                # Revenue KPIs
                revenue_kpis = pd.DataFrame({
                    'Metric': ['Total Revenue', 'Average Daily Revenue', 'Revenue Volatility', 'Revenue Skewness',
                             'Peak Revenue', 'Revenue Consistency', 'Month-over-Month Growth'],
                    'Value': [total_revenue, avg_daily_revenue, revenue_volatility, revenue_skewness,
                             peak_revenue, revenue_consistency, mom_growth]
                })
                revenue_kpis.to_excel(writer, sheet_name='Revenue KPIs', index=False)
                
            output.seek(0)
            st.download_button(
                label="📥 Download Revenue Report",
                data=output,
                file_name=f"revenue_detailed_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with report_cols[1]:
        if st.button("Generate Payment Analysis Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Daily Payment Mix
                payment_mix = filtered_data["daily_income"][["date", "cash", "visa", "due amount", "Total"]].copy()
                payment_mix[["cash_pct", "visa_pct", "due_pct"]] = payment_mix[["cash", "visa", "due amount"]].div(payment_mix["Total"], axis=0) * 100
                payment_mix.round(2).to_excel(writer, sheet_name='Daily Payment Mix')
                
                # Payment Method Statistics
                payment_stats = filtered_data["daily_income"][["cash", "visa", "due amount"]].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).round(2)
                payment_stats.to_excel(writer, sheet_name='Payment Statistics')
                
            output.seek(0)
            st.download_button(
                label="📥 Download Payment Analysis",
                data=output,
                file_name=f"payment_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with report_cols[2]:
        if st.button("Generate Growth Analysis Report"):
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Weekly Growth
                weekly_growth = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.strftime('%Y-%W')
                ).agg({
                    'Total': ['sum', 'mean', 'std'],
                    'cash': 'sum',
                    'visa': 'sum',
                    'due amount': 'sum'
                }).round(2)
                weekly_growth.to_excel(writer, sheet_name='Weekly Growth')
                
                # Day of Week Analysis
                dow_analysis = filtered_data["daily_income"].groupby(
                    filtered_data["daily_income"]["date"].dt.day_name()
                ).agg({
                    'Total': ['count', 'sum', 'mean', 'std'],
                    'cash': ['sum', 'mean'],
                    'visa': ['sum', 'mean'],
                    'due amount': ['sum', 'mean']
                }).round(2)
                dow_analysis.to_excel(writer, sheet_name='Day of Week Analysis')
                
            output.seek(0)
            st.download_button(
                label="📥 Download Growth Analysis",
                data=output,
                file_name=f"growth_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

# Inventory Tab
with tab_inventory:
    st.markdown("### 📦 Inventory Management Dashboard")
    
    # First Row - Main KPIs
    st.markdown("#### 📊 Primary Metrics")
    kpi_cols = st.columns(5)
    
    # Calculate main KPIs
    total_purchases = filtered_data["inventory"]["Invoice Amount"].sum()
    total_credit = filtered_data["inventory"]["Credit Limit"].sum()
    avg_invoice = filtered_data["inventory"]["Invoice Amount"].mean()
    num_suppliers = filtered_data["inventory"]["Invoice Company"].nunique()
    inventory_types_count = filtered_data["inventory"]["Inventory Type"].nunique()

    with kpi_cols[0]:
        st.metric("Total Purchases", f"EGP {total_purchases:,.2f}")
    with kpi_cols[1]:
        st.metric("Total Credit Limit", f"EGP {total_credit:,.2f}")
    with kpi_cols[2]:
        st.metric("Average Invoice", f"EGP {avg_invoice:,.2f}")
    with kpi_cols[3]:
        st.metric("Active Suppliers", f"{num_suppliers}")
    with kpi_cols[4]:
        st.metric("Inventory Categories", f"{inventory_types_count}")

    # Second Row - Credit Analysis
    st.markdown("#### 💳 Credit Management")
    credit_cols = st.columns(4)
    
    # Calculate credit metrics
    credit_utilization = (total_purchases / total_credit * 100) if total_credit > 0 else 0
    avg_credit_limit = filtered_data["inventory"]["Credit Limit"].mean()
    max_credit = filtered_data["inventory"]["Credit Limit"].max()
    
    with credit_cols[0]:
        st.metric("Credit Utilization", f"{credit_utilization:.1f}%")
    with credit_cols[1]:
        st.metric("Average Credit Limit", f"EGP {avg_credit_limit:,.2f}")
    with credit_cols[2]:
        st.metric("Maximum Credit Line", f"EGP {max_credit:,.2f}")
    with credit_cols[3]:
        st.metric("Credit to Purchase Ratio", 
                 f"{(total_credit/total_purchases if total_purchases > 0 else 0):.2f}x")

    # Third Row - Supplier Analysis
    st.markdown("#### 🏢 Supplier Performance")
    supplier_cols = st.columns(2)
    
    with supplier_cols[0]:
        # Top Suppliers by Volume
        supplier_volume = filtered_data["inventory"].groupby("Invoice Company").agg({
            "Invoice Amount": "sum",
            "id": "count"
        }).sort_values("Invoice Amount", ascending=False)
        
        fig_suppliers = go.Figure()
        fig_suppliers.add_trace(go.Bar(
            x=supplier_volume.head(10).index,
            y=supplier_volume.head(10)["Invoice Amount"],
            name="Purchase Volume",
            marker_color=COLOR_PALETTE["primary"]
        ))
        fig_suppliers.add_trace(go.Scatter(
            x=supplier_volume.head(10).index,
            y=supplier_volume.head(10)["id"],
            name="Number of Invoices",
            yaxis="y2",
            line=dict(color=COLOR_PALETTE["accent"])
        ))
        fig_suppliers.update_layout(
            title="Top 10 Suppliers by Purchase Volume",
            yaxis=dict(title="Purchase Amount (EGP)"),
            yaxis2=dict(title="Number of Invoices", overlaying="y", side="right"),
            showlegend=True,
            height=400
        )
        st.plotly_chart(fig_suppliers, use_container_width=True)
    
    with supplier_cols[1]:
        # Credit Limit Distribution
        credit_dist = filtered_data["inventory"].groupby("Invoice Company")["Credit Limit"].mean()
        fig_credit = go.Figure(data=[go.Bar(
            x=credit_dist.sort_values(ascending=False).head(10).index,
            y=credit_dist.sort_values(ascending=False).head(10).values,
            marker_color=COLOR_PALETTE["secondary"]
        )])
        fig_credit.update_layout(
            title="Top 10 Companies by Credit Limit",
            xaxis_title="Company",
            yaxis_title="Credit Limit (EGP)",
            height=400
        )
        st.plotly_chart(fig_credit, use_container_width=True)

    # Fourth Row - Inventory Type Analysis
    st.markdown("#### 📦 Inventory Categories")
    type_cols = st.columns(2)
    
    with type_cols[0]:
        # Inventory Type Distribution
        type_dist = filtered_data["inventory"].groupby("Inventory Type").agg({
            "Invoice Amount": "sum",
            "id": "count"
        })
        
        fig_types = go.Figure(data=[go.Pie(
            labels=type_dist.index,
            values=type_dist["Invoice Amount"],
            hole=0.4,
            textinfo="label+percent"
        )])
        fig_types.update_layout(
            title="Purchase Distribution by Inventory Type",
            height=400
        )
        st.plotly_chart(fig_types, use_container_width=True)
    
    with type_cols[1]:
        # Type Trends Over Time
        type_trends = filtered_data["inventory"].groupby([
            filtered_data["inventory"]["date"].dt.strftime("%Y-%m"),  # Convert to string format instead of Period
            "Inventory Type"
        ])["Invoice Amount"].sum().reset_index()
        type_trends.columns = ["date", "Inventory Type", "Invoice Amount"]  # Rename columns for clarity
        
        fig_trends = px.line(
            type_trends,
            x="date",
            y="Invoice Amount",
            color="Inventory Type",
            title="Monthly Trends by Inventory Type"
        )
        fig_trends.update_layout(
            height=400,
            xaxis_title="Month",
            yaxis_title="Amount (EGP)",
            template="plotly_white"
        )
        st.plotly_chart(fig_trends, use_container_width=True)

    # Fifth Row - Invoice Analysis
    st.markdown("#### 📝 Invoice Analytics")
    invoice_cols = st.columns(3)
    
    with invoice_cols[0]:
        # Invoice Size Distribution
        fig_invoice_dist = px.histogram(
            filtered_data["inventory"],
            x="Invoice Amount",
            nbins=50,
            title="Invoice Amount Distribution"
        )
        fig_invoice_dist.update_layout(height=300)
        st.plotly_chart(fig_invoice_dist, use_container_width=True)
    
    with invoice_cols[1]:
        # Daily Invoice Count
        daily_invoices = filtered_data["inventory"].groupby("date")["id"].count()
        fig_daily = go.Figure(data=[go.Scatter(
            x=daily_invoices.index,
            y=daily_invoices.values,
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE["primary"])
        )])
        fig_daily.update_layout(
            title="Daily Invoice Count",
            height=300
        )
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with invoice_cols[2]:
        # Invoice Type Distribution
        invoice_types = filtered_data["inventory"].groupby("Invoice Type")["Invoice Amount"].sum()
        fig_inv_types = go.Figure(data=[go.Pie(
            labels=invoice_types.index,
            values=invoice_types.values,
            hole=0.4
        )])
        fig_inv_types.update_layout(
            title="Distribution by Invoice Type",
            height=300
        )
        st.plotly_chart(fig_inv_types, use_container_width=True)

    # Sixth Row - Detailed Analysis
    st.markdown("#### 📊 Detailed Metrics")
    detail_cols = st.columns(4)
    
    # Calculate additional metrics
    avg_daily_purchase = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum().mean()
    purchase_std = filtered_data["inventory"]["Invoice Amount"].std()
    largest_invoice = filtered_data["inventory"]["Invoice Amount"].max()
    invoice_count = len(filtered_data["inventory"])
    
    with detail_cols[0]:
        st.metric("Avg Daily Purchase", f"EGP {avg_daily_purchase:,.2f}")
    with detail_cols[1]:
        st.metric("Purchase Std Dev", f"EGP {purchase_std:,.2f}")
    with detail_cols[2]:
        st.metric("Largest Invoice", f"EGP {largest_invoice:,.2f}")
    with detail_cols[3]:
        st.metric("Total Invoices", f"{invoice_count:,}")

    # Seventh Row - Data Table
    st.markdown("#### 📋 Detailed Purchase Records")
    
    # Create expandable detailed view
    with st.expander("View Detailed Purchase Records"):
        # Add search functionality
        search_term = st.text_input("Search by Invoice ID or Company")
        
        if search_term:
            filtered_view = filtered_data["inventory"][
                filtered_data["inventory"].astype(str).apply(
                    lambda x: x.str.contains(search_term, case=False)
                ).any(axis=1)
            ]
        else:
            filtered_view = filtered_data["inventory"]
        
        st.dataframe(
            filtered_view.sort_values("date", ascending=False),
            use_container_width=True
        )

# Expenses Tab
with tab_expenses:
    st.markdown("### 💸 Expense Analysis")
    
    # Expense KPIs
    st.markdown("#### 📊 Expense KPIs")
    expense_kpi_cols = st.columns(4)
    
    with expense_kpi_cols[0]:
        total_expenses = filtered_data["expenses"]["Expense Amount"].sum()
        st.metric("Total Expenses", f"EGP {total_expenses:,.2f}")
    
    with expense_kpi_cols[1]:
        avg_expense = filtered_data["expenses"]["Expense Amount"].mean()
        st.metric("Average Expense", f"EGP {avg_expense:,.2f}")
    
    with expense_kpi_cols[2]:
        expense_types = filtered_data["expenses"]["Expense Type"].nunique()
        st.metric("Expense Types", f"{expense_types}")
    
    with expense_kpi_cols[3]:
        expense_ratio = (total_expenses / total_income * 100) if total_income > 0 else 0
        st.metric("Expense Ratio", f"{expense_ratio:.1f}%")
    
    # Expense Distribution
    expense_dist = filtered_data["expenses"].groupby("Expense Type")["Expense Amount"].sum().sort_values(ascending=True)
    colors = px.colors.qualitative.Set3[:len(expense_dist)]

    fig_expense = go.Figure(data=[go.Pie(
        labels=expense_dist.index,
        values=expense_dist.values,
        hole=.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside'
    )])
    fig_expense.update_layout(
        title="Expense Distribution by Type",
        template="plotly_white",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    st.plotly_chart(fig_expense, use_container_width=True)

    # Monthly Expense Trend by Type
    monthly_expenses = filtered_data["expenses"].groupby([
        filtered_data["expenses"]["date"].dt.to_period("M"),
        "Expense Type"
    ])["Expense Amount"].sum().reset_index()
    monthly_expenses["date"] = monthly_expenses["date"].astype(str)

    fig_monthly_exp = px.bar(
        monthly_expenses,
        x="date",
        y="Expense Amount",
        color="Expense Type",
        title="Monthly Expense Trend by Type",
        labels={"date": "Month", "Expense Amount": "Amount (EGP)"},
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_monthly_exp.update_layout(
        template="plotly_white",
        xaxis_title="Month",
        yaxis_title="Amount (EGP)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=400
    )
    st.plotly_chart(fig_monthly_exp, use_container_width=True)

# Analytics Tab
with tab_analytics:
    st.markdown("### 📈 Advanced Analytics")
    
    # Analytics KPIs
    st.markdown("#### 📊 Analytics KPIs")
    analytics_kpi_cols = st.columns(4)
    
    with analytics_kpi_cols[0]:
        profit_margin = (net_profit / total_income * 100) if total_income > 0 else 0
        st.metric("Profit Margin", f"{profit_margin:.1f}%")
    
    with analytics_kpi_cols[1]:
        inventory_turnover = total_income / total_purchases if total_purchases > 0 else 0
        st.metric("Inventory Turnover", f"{inventory_turnover:.2f}x")
    
    with analytics_kpi_cols[2]:
        avg_transaction = filtered_data["daily_income"]["Total"].mean()
        st.metric("Avg Transaction", f"EGP {avg_transaction:,.2f}")
    
    with analytics_kpi_cols[3]:
        data_points = len(filtered_data["daily_income"]) + len(filtered_data["inventory"]) + len(filtered_data["expenses"])
        st.metric("Total Data Points", f"{data_points:,}")
    
    # Data Tables
    st.markdown("#### 📋 Detailed Data Tables")
    tab1, tab2, tab3 = st.tabs(["Daily Income", "Inventory", "Expenses"])
    
    with tab1:
        st.dataframe(
            filtered_data["daily_income"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab2:
        st.dataframe(
            filtered_data["inventory"].sort_values("date", ascending=False),
            use_container_width=True
        )
    
    with tab3:
        st.dataframe(
            filtered_data["expenses"].sort_values("date", ascending=False),
            use_container_width=True
        )

# ML & Predictions Tab
with tab_ml:
    st.markdown("### 🤖 Advanced Analytics & Predictions")
    
    # Setup prediction data
    pred_cols = st.columns([2, 1])
    with pred_cols[1]:
        prediction_days = st.slider("Prediction Days", 7, 90, 30)
        confidence_interval = st.slider("Confidence Interval", 0.8, 0.99, 0.95)
        
    # Prepare integrated dataset for predictions
    daily_metrics = pd.DataFrame()
    daily_metrics['date'] = filtered_data["daily_income"]["date"]
    daily_metrics['revenue'] = filtered_data["daily_income"]["Total"]
    
    # Ensure expenses and purchases are aligned with revenue dates
    expenses_agg = filtered_data["expenses"].groupby("date")["Expense Amount"].sum().reindex(daily_metrics['date']).fillna(0)
    purchases_agg = filtered_data["inventory"].groupby("date")["Invoice Amount"].sum().reindex(daily_metrics['date']).fillna(0)
    
    daily_metrics['expenses'] = expenses_agg.values
    daily_metrics['purchases'] = purchases_agg.values
    daily_metrics['deficit'] = filtered_data["daily_income"]["deficit"]
    daily_metrics['cash'] = filtered_data["daily_income"]["cash"]
    daily_metrics['visa'] = filtered_data["daily_income"]["visa"]
    daily_metrics['due_amount'] = filtered_data["daily_income"]["due amount"]
    daily_metrics = daily_metrics.fillna(0)
    
    # Calculate derived metrics
    daily_metrics['net_profit'] = daily_metrics['revenue'] - daily_metrics['expenses'] - daily_metrics['purchases']
    daily_metrics['profit_margin'] = (daily_metrics['net_profit'] / daily_metrics['revenue']).fillna(0)
    daily_metrics['expense_ratio'] = (daily_metrics['expenses'] / daily_metrics['revenue']).fillna(0)
    
    # Multi-metric Prophet Models
    metrics_to_predict = {
        'Revenue': daily_metrics['revenue'],
        'Expenses': daily_metrics['expenses'],
        'Purchases': daily_metrics['purchases'],
        'Net Profit': daily_metrics['net_profit']
    }
    
    forecast_results = {}
    model_metrics = {}
    
    with st.spinner("Training multiple prediction models..."):
        for metric_name, metric_data in metrics_to_predict.items():
            # Prepare data
            df_prophet = pd.DataFrame({
                'ds': daily_metrics['date'],
                'y': metric_data
            })
            
            # Configure model
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=True,
                changepoint_prior_scale=0.05,
                interval_width=confidence_interval
            )
            
            # Add custom seasonality
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit model
            model.fit(df_prophet)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=prediction_days)
            forecast = model.predict(future)
            
            forecast_results[metric_name] = {
                'model': model,
                'forecast': forecast,
                'actual': df_prophet
            }
            
            # Calculate model metrics
            train_rmse = np.sqrt(mean_squared_error(
                df_prophet['y'],
                forecast['yhat'][:len(df_prophet)]
            ))
            model_metrics[metric_name] = {
                'rmse': train_rmse,
                'accuracy': 1 - (train_rmse / df_prophet['y'].mean()) if df_prophet['y'].mean() != 0 else np.nan
            }
    
    # Display Model Performance Metrics
    st.markdown("#### 📊 Model Performance")
    metric_cols = st.columns(len(model_metrics))
    for idx, (metric_name, metrics) in enumerate(model_metrics.items()):
        with metric_cols[idx]:
            st.metric(
                f"{metric_name} Model Accuracy",
                f"{metrics['accuracy']*100:.1f}%" if not np.isnan(metrics['accuracy']) else "N/A",
                f"RMSE: {metrics['rmse']:,.2f}"
            )
    
    # Integrated Forecast Visualization
    st.markdown("#### 📈 Integrated Forecasts")
    
    forecast_tabs = st.tabs([
        "Revenue & Profit",
        "Expenses & Purchases",
        "Seasonality Analysis",
        "Correlation Analysis"
    ])
    
    with forecast_tabs[0]:
        # Revenue and Profit Forecasts
        fig = go.Figure()
        
        for metric in ['Revenue', 'Net Profit']:
            forecast = forecast_results[metric]['forecast']
            actual = forecast_results[metric]['actual']
            
            # Actual values
            fig.add_trace(go.Scatter(
                x=actual['ds'],
                y=actual['y'],
                name=f'Actual {metric}',
                line=dict(color=COLOR_PALETTE['primary'] if metric == 'Revenue' else COLOR_PALETTE['accent'])
            ))
            
            # Forecast values
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'Forecast {metric}',
                line=dict(dash='dash', color=COLOR_PALETTE['primary'] if metric == 'Revenue' else COLOR_PALETTE['accent'])
            ))
            
            # Confidence intervals
            fig.add_trace(go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor=f'rgba{tuple(list(int(COLOR_PALETTE["primary"].lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{metric} Confidence Interval'
            ))
        
        fig.update_layout(
            title="Revenue and Profit Forecast",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Key Metrics
        metric_cols = st.columns(4)
        for idx, metric in enumerate(['Revenue', 'Net Profit']):
            forecast = forecast_results[metric]['forecast']
            last_actual = forecast_results[metric]['actual']['y'].iloc[-1]
            last_forecast = forecast['yhat'].iloc[-1]
            
            with metric_cols[idx*2]:
                growth_rate = ((last_forecast - last_actual) / last_actual) * 100
                st.metric(
                    f"{metric} Growth",
                    f"{growth_rate:.1f}%",
                    f"EGP {last_forecast - last_actual:,.2f}"
                )
            
            with metric_cols[idx*2 + 1]:
                forecast_avg = forecast['yhat'].tail(prediction_days).mean()
                current_avg = forecast_results[metric]['actual']['y'].tail(prediction_days).mean()
                st.metric(
                    f"{metric} Trend",
                    f"EGP {forecast_avg:,.2f}",
                    f"{((forecast_avg - current_avg) / current_avg * 100):.1f}%"
                )
    
    with forecast_tabs[1]:
        # Expenses and Purchases Forecasts
        fig = go.Figure()
        
        for metric in ['Expenses', 'Purchases']:
            forecast = forecast_results[metric]['forecast']
            actual = forecast_results[metric]['actual']
            
            fig.add_trace(go.Scatter(
                x=actual['ds'],
                y=actual['y'],
                name=f'Actual {metric}',
                line=dict(color=COLOR_PALETTE['secondary'] if metric == 'Expenses' else COLOR_PALETTE['neutral'])
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                name=f'Forecast {metric}',
                line=dict(dash='dash', color=COLOR_PALETTE['secondary'] if metric == 'Expenses' else COLOR_PALETTE['neutral'])
            ))
        
        fig.update_layout(
            title="Expenses and Purchases Forecast",
            xaxis_title="Date",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast Impact Analysis
        impact_cols = st.columns(2)
        
        with impact_cols[0]:
            expenses_forecast = forecast_results['Expenses']['forecast']
            purchases_forecast = forecast_results['Purchases']['forecast']
            
            total_cost_forecast = expenses_forecast['yhat'].tail(prediction_days).sum() + \
                                purchases_forecast['yhat'].tail(prediction_days).sum()
            
            current_total_cost = daily_metrics['expenses'].tail(prediction_days).sum() + \
                               daily_metrics['purchases'].tail(prediction_days).sum()
            
            st.metric(
                "Projected Cost Impact",
                f"EGP {total_cost_forecast:,.2f}",
                f"{((total_cost_forecast - current_total_cost) / current_total_cost * 100):.1f}%"
            )
        
        with impact_cols[1]:
            cost_ratio_forecast = total_cost_forecast / forecast_results['Revenue']['forecast']['yhat'].tail(prediction_days).sum()
            current_cost_ratio = (daily_metrics['expenses'].tail(prediction_days).sum() + 
                                daily_metrics['purchases'].tail(prediction_days).sum()) / \
                                daily_metrics['revenue'].tail(prediction_days).sum()
            
            st.metric(
                "Projected Cost Ratio",
                f"{cost_ratio_forecast:.1%}",
                f"{(cost_ratio_forecast - current_cost_ratio) * 100:.1f}%"
            )
    
    with forecast_tabs[2]:
        # Seasonality Analysis
        for metric_name, metric_results in forecast_results.items():
            model = metric_results['model']
            
            st.markdown(f"##### {metric_name} Seasonality Components")
            
            # Plot seasonality components
            fig = model.plot_components(metric_results['forecast'])
            st.pyplot(fig)
            plt.close()
            
            # Create mapping for metric names to column names
            metric_mapping = {
                'Revenue': 'revenue',
                'Expenses': 'expenses',
                'Purchases': 'purchases',
                'Net Profit': 'net_profit'
            }
            
            # Weekly patterns using correct column names
            metric_col = metric_mapping[metric_name]
            weekly_pattern = daily_metrics.groupby(daily_metrics['date'].dt.day_name())[metric_col].mean()
            
            fig = go.Figure(data=[go.Bar(
                x=weekly_pattern.index,
                y=weekly_pattern.values,
                marker_color=COLOR_PALETTE['primary']
            )])
            
            fig.update_layout(
                title=f"{metric_name} - Average by Day of Week",
                xaxis_title="Day of Week",
                yaxis_title="Amount (EGP)",
                template="plotly_white",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with forecast_tabs[3]:
        # Correlation Analysis
        st.markdown("##### 🔄 Metric Correlations")
        
        # Calculate correlations
        correlation_matrix = daily_metrics[['revenue', 'expenses', 'purchases', 'net_profit']].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Cross-correlation analysis
        lag_range = st.slider("Lag Range (Days)", -30, 30, (-7, 7))
        
        reference_metric = st.selectbox(
            "Reference Metric",
            options=['revenue', 'expenses', 'purchases', 'net_profit']
        )
        
        for metric in ['revenue', 'expenses', 'purchases', 'net_profit']:
            if metric != reference_metric:
                ccf = pd.DataFrame(
                    index=range(lag_range[0], lag_range[1] + 1),
                    columns=['correlation']
                )
                
                for lag in range(lag_range[0], lag_range[1] + 1):
                    if lag < 0:
                        ccf.loc[lag, 'correlation'] = daily_metrics[reference_metric].corr(
                            daily_metrics[metric].shift(-lag)
                        )
                    else:
                        ccf.loc[lag, 'correlation'] = daily_metrics[reference_metric].corr(
                            daily_metrics[metric].shift(lag)
                        )
                
                fig = go.Figure(data=go.Bar(
                    x=ccf.index,
                    y=ccf['correlation'],
                    marker_color=COLOR_PALETTE['primary']
                ))
                
                fig.update_layout(
                    title=f"Cross-correlation: {reference_metric.title()} vs {metric.title()}",
                    xaxis_title="Lag (Days)",
                    yaxis_title="Correlation",
                    template="plotly_white",
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)

# Search & Reports Tab
with tab_search:
    st.markdown("### 🔍 Inventory Purchase Search & Reports")
    
    # Search KPIs
    st.markdown("#### 📊 Search Results Summary")
    search_kpi_cols = st.columns(4)
    
    with search_kpi_cols[0]:
        total_purchases = len(filtered_data["inventory"])
        st.metric("Total Purchases", f"{total_purchases:,}")
    
    with search_kpi_cols[1]:
        total_amount = filtered_data["inventory"]["Invoice Amount"].sum()
        st.metric("Total Amount", f"EGP {total_amount:,.2f}")
    
    with search_kpi_cols[2]:
        avg_purchase = filtered_data["inventory"]["Invoice Amount"].mean()
        st.metric("Average Purchase", f"EGP {avg_purchase:,.2f}")
    
    with search_kpi_cols[3]:
        unique_companies = filtered_data["inventory"]["Invoice Company"].nunique()
        st.metric("Unique Companies", f"{unique_companies}")
    
    # Search Filters
    st.markdown("#### 🔎 Search Filters")
    search_cols = st.columns(5)
    
    with search_cols[0]:
        invoice_id = st.text_input(
            "Invoice ID",
            placeholder="Enter Invoice ID",
            help="Search by specific invoice ID"
        )
    
    with search_cols[1]:
        search_company = st.selectbox(
            "Company",
            ["All"] + sorted(filtered_data["inventory"]["Invoice Company"].unique().tolist())
        )
    
    with search_cols[2]:
        search_type = st.selectbox(
            "Inventory Type",
            ["All"] + sorted(filtered_data["inventory"]["Inventory Type"].unique().tolist())
        )
    
    with search_cols[3]:
        min_amount = st.number_input(
            "Min Amount (EGP)",
            min_value=0.0,
            max_value=float(filtered_data["inventory"]["Invoice Amount"].max()),
            value=0.0
        )
    
    with search_cols[4]:
        max_amount = st.number_input(
            "Max Amount (EGP)",
            min_value=0.0,
            max_value=float(filtered_data["inventory"]["Invoice Amount"].max()),
            value=float(filtered_data["inventory"]["Invoice Amount"].max())
        )
    
    # Apply filters
    search_results = filtered_data["inventory"].copy()
    
    # Apply Invoice ID filter if provided
    if invoice_id:
        search_results = search_results[
            search_results["Invoice ID"].astype(str).str.contains(invoice_id, case=False, na=False)
        ]
    
    if search_company != "All":
        search_results = search_results[search_results["Invoice Company"] == search_company]
    
    if search_type != "All":
        search_results = search_results[search_results["Inventory Type"] == search_type]
    
    search_results = search_results[
        (search_results["Invoice Amount"] >= min_amount) &
        (search_results["Invoice Amount"] <= max_amount)
    ]
    
    # Display search results
    st.markdown("#### 📋 Search Results")
    
    # Results summary
    results_cols = st.columns(3)
    
    with results_cols[0]:
        st.metric(
            "Filtered Purchases",
            f"{len(search_results):,}",
            delta=f"{len(search_results) - len(filtered_data['inventory']):,}"
        )
    
    with results_cols[1]:
        filtered_amount = search_results["Invoice Amount"].sum()
        st.metric(
            "Filtered Amount",
            f"EGP {filtered_amount:,.2f}",
            delta=f"EGP {filtered_amount - total_amount:,.2f}"
        )
    
    with results_cols[2]:
        filtered_avg = search_results["Invoice Amount"].mean()
        st.metric(
            "Filtered Average",
            f"EGP {filtered_avg:,.2f}",
            delta=f"EGP {filtered_avg - avg_purchase:,.2f}"
        )
    
    # Detailed results table
    st.dataframe(
        search_results.sort_values("date", ascending=False),
        use_container_width=True
    )
    
    # Purchase Analysis
    st.markdown("#### 📊 Purchase Analysis")
    analysis_cols = st.columns(2)
    
    with analysis_cols[0]:
        # Company distribution
        company_dist = search_results.groupby("Invoice Company")["Invoice Amount"].sum().sort_values(ascending=True)
        fig_company = go.Figure(data=[go.Bar(
            x=company_dist.values,
            y=company_dist.index,
            orientation='h',
            marker_color=COLOR_PALETTE["primary"]
        )])
        fig_company.update_layout(
            title="Purchases by Company",
            xaxis_title="Amount (EGP)",
            yaxis_title="Company",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_company, use_container_width=True)
    
    with analysis_cols[1]:
        # Monthly trend
        monthly_purchases = search_results.groupby(
            search_results["date"].dt.to_period("M")
        )["Invoice Amount"].sum()
        
        fig_monthly = go.Figure(data=[go.Scatter(
            x=monthly_purchases.index.astype(str),
            y=monthly_purchases.values,
            mode='lines+markers',
            line=dict(color=COLOR_PALETTE["secondary"])
        )])
        fig_monthly.update_layout(
            title="Monthly Purchase Trend",
            xaxis_title="Month",
            yaxis_title="Amount (EGP)",
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_monthly, use_container_width=True)
    
    # Export functionality
    st.markdown("#### 📥 Export Results")
    export_cols = st.columns(2)
    
    with export_cols[0]:
        if st.button("Export to Excel"):
            # Create Excel writer
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Write search results
                search_results.to_excel(writer, sheet_name='Search Results', index=False)
                
                # Write summary
                summary_data = pd.DataFrame({
                    'Metric': ['Total Purchases', 'Total Amount', 'Average Purchase', 'Unique Companies'],
                    'Value': [len(search_results), filtered_amount, filtered_avg, search_results["Invoice Company"].nunique()]
                })
                summary_data.to_excel(writer, sheet_name='Summary', index=False)
            
            # Create download button
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output,
                file_name=f"inventory_purchase_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with export_cols[1]:
        if st.button("Generate PDF Report"):
            st.info("PDF report generation will be implemented in the next version.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Pharmacy Analytics Dashboard • Updated: "
    f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>", 
    unsafe_allow_html=True
)
