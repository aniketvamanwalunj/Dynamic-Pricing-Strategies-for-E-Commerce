import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------

st.set_page_config(
    page_title="Dynamic Pricing Strategies for E-Commerce",
    layout="wide"
)

# -----------------------------------------------------
# SIDEBAR LOGO (BIG SIZE)
# -----------------------------------------------------

# This places the logo at the very top of the sidebar.
# use_container_width=True ensures it takes up the full width of the sidebar [web:2].
try:
    st.sidebar.image("logo.png", use_container_width=True)
except Exception:
    st.sidebar.error("logo.png not found. Please place it in the app directory.")

st.title("Dynamic Pricing Strategies for E-Commerce")
st.subheader("Myntra Fashion E-Commerce Dataset")

# -----------------------------------------------------
# LOAD DATA
# -----------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("Myntra_Clean_Data.csv")
    df["date"] = pd.to_datetime(df["date"])

    # numeric cleanup
    num_cols = [
        "listed_price", "final_price", "cost_price",
        "discount_pct", "units_sold", "inventory_level",
        "competitor_price", "revenue", "profit"
    ]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # -------------------------------
    # CREATE season & day_of_week
    # -------------------------------

    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Summer"
        elif month in [6, 7, 8]:
            return "Monsoon"
        else:
            return "Festive"

    df["season"] = df["date"].dt.month.apply(get_season)
    df["day_of_week"] = df["date"].dt.day_name()

    return df

df = load_data()

# -----------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------

st.sidebar.header("App Navigation")

page = st.sidebar.radio(
    "Go To Page:",
    [
        "Dashboard Overview",
        "Data Quality Checks",
        "Demand Prediction",
        "Time Series Forecasting",
        "Pricing & Revenue Insights"
    ]
)

# -----------------------------------------------------
# GLOBAL FILTERS
# -----------------------------------------------------

st.sidebar.header("Global Filters")

selected_categories = st.sidebar.multiselect(
    "Product Category",
    df["category"].unique(),
    default=df["category"].unique()
)

selected_seasons = st.sidebar.multiselect(
    "Season",
    df["season"].unique(),
    default=df["season"].unique()
)

selected_days = st.sidebar.multiselect(
    "Day of Week",
    df["day_of_week"].unique(),
    default=df["day_of_week"].unique()
)

start_date, end_date = st.sidebar.date_input(
    "Date Range",
    [
        df["date"].min(),
        df["date"].max()
    ]
)

selected_products = st.sidebar.multiselect(
    "Product ID (Optional)",
    df["product_id"].unique()
)

# -----------------------------------------------------
# APPLY FILTERS
# -----------------------------------------------------

filtered_df = df[
    (df["category"].isin(selected_categories)) &
    (df["season"].isin(selected_seasons)) &
    (df["day_of_week"].isin(selected_days)) &
    (df["date"].between(
        pd.to_datetime(start_date),
        pd.to_datetime(end_date)
    ))
]

if selected_products:
    filtered_df = filtered_df[
        filtered_df["product_id"].isin(selected_products)
    ]

# -----------------------------------------------------
# DOWNLOAD FILTERED DATA (GLOBAL)
# -----------------------------------------------------

st.sidebar.markdown("Download")
csv_download = filtered_df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download Filtered Data (CSV)",
    data=csv_download,
    file_name="myntra_filtered_data.csv",
    mime="text/csv"
)

# -----------------------------------------------------
# KPI FUNCTION
# -----------------------------------------------------

def show_kpis(df_in):
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Revenue", f"{df_in['revenue'].sum():,.0f}")
    c2.metric("Total Units Sold", int(df_in["units_sold"].sum()))
    c3.metric("Average Final Price", f"{df_in['final_price'].mean():.2f}")
    c4.metric("Average Discount %", f"{df_in['discount_pct'].mean():.2f}")
    c5.metric("Average Inventory", int(df_in["inventory_level"].mean()))

# =====================================================
# ================= DASHBOARD OVERVIEW ===============
# =====================================================

if page == "Dashboard Overview":

    st.header("Descriptive Dashboard (EDA)")
    show_kpis(filtered_df)

    st.subheader("Time and Trend")

    fig = px.line(
        filtered_df.groupby("date")["units_sold"].sum().reset_index(),
        x="date",
        y="units_sold",
        title="Date vs Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(
        filtered_df.groupby("date")["revenue"].sum().reset_index(),
        x="date",
        y="revenue",
        title="Date vs Revenue"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Seasonality and Weekly Patterns")

    fig = px.bar(
        filtered_df.groupby("season")["units_sold"].mean().reset_index(),
        x="season",
        y="units_sold",
        title="Season vs Average Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(
        filtered_df.groupby("day_of_week")["units_sold"].mean().reset_index(),
        x="day_of_week",
        y="units_sold",
        title="Day of Week vs Average Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    heatmap_df = filtered_df.pivot_table(
        values="units_sold",
        index="day_of_week",
        columns="season",
        aggfunc="mean"
    )

    fig = px.imshow(heatmap_df, title="Heatmap: Day vs Season")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Product and Pricing")

    fig = px.bar(
        filtered_df.groupby("category")["units_sold"].sum().reset_index(),
        x="category",
        y="units_sold",
        title="Category vs Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered_df,
        x="final_price",
        y="units_sold",
        title="Final Price vs Units Sold"
    )
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(
        filtered_df,
        x="final_price",
        y="competitor_price",
        title="Final Price vs Competitor Price"
    )
    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# =============== DATA QUALITY CHECKS PAGE ============
# =====================================================

elif page == "Data Quality Checks":

    st.header("Data Quality Checks")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("Missing Values (Top 10 Columns)")
        missing = filtered_df.isna().sum().sort_values(ascending=False)
        st.dataframe(missing.head(10))

        st.markdown("Duplicate Rows")
        dup_count = filtered_df.duplicated().sum()
        st.write(f"Total duplicate rows: {dup_count}")

    with col_b:
        st.markdown("Numeric Summary (Key Metrics)")
        st.dataframe(
            filtered_df[
                ["listed_price", "final_price", "cost_price",
                 "discount_pct", "units_sold", "inventory_level",
                 "competitor_price", "revenue", "profit"]
            ].describe().T
        )

# =====================================================
# ================= DEMAND PREDICTION ================
# =====================================================

elif page == "Demand Prediction":

    st.header("Demand Prediction (Regression)")

    model_df = df.copy()

    le_season = LabelEncoder()
    le_day = LabelEncoder()

    model_df["season_enc"] = le_season.fit_transform(model_df["season"])
    model_df["day_enc"] = le_day.fit_transform(model_df["day_of_week"])

    features = [
        "final_price",
        "discount_pct",
        "competitor_price",
        "inventory_level",
        "season_enc",
        "day_enc"
    ]

    X = model_df[features]
    y = model_df["units_sold"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ---------------- PREDICTION INPUTS ----------------
    st.subheader("Prediction Inputs")

    col1, col2, col3 = st.columns(3)

    price = col1.number_input("Final Price", 100.0, 20000.0, 1500.0)
    discount = col2.slider("Discount %", 0.0, 80.0, 10.0)
    comp_price = col3.number_input("Competitor Price", 100.0, 20000.0, 1400.0)

    inventory = st.slider("Inventory Level", 10, 10000, 500)

    season_val = st.selectbox("Season", le_season.classes_)
    day_val = st.selectbox("Day of Week", le_day.classes_)

    cost = st.number_input("Cost Price", 50.0, 15000.0, 900.0)

    season_enc = le_season.transform([season_val])[0]
    day_enc = le_day.transform([day_val])[0]

    if st.button("Predict Demand"):

        pred = model.predict(
            [[price, discount, comp_price,
              inventory, season_enc, day_enc]]
        )[0]

        revenue = price * pred
        profit = (price - cost) * pred

        st.success(f"Predicted Units Sold: {int(pred)}")
        st.info(f"Expected Revenue: {revenue:,.2f}")
        st.warning(f"Expected Profit: {profit:,.2f}")

# =====================================================
# ================= TIME SERIES ======================
# =====================================================

elif page == "Time Series Forecasting":

    st.header("Demand Forecasting")

    daily_demand = (
        filtered_df.groupby("date")["units_sold"]
        .sum()
        .reset_index()
        .sort_values("date")
    )

    fig = px.line(
        daily_demand,
        x="date",
        y="units_sold",
        title="Historical Demand"
    )
    st.plotly_chart(fig, use_container_width=True)

    if daily_demand.empty:
        st.warning("No data available for the selected filters to build a time series.")
    else:
        ts = daily_demand.set_index("date")["units_sold"]

        st.subheader("Forecast Settings")

        model_type = st.radio(
            "Select Time Series Model",
            ("ARIMA", "SARIMA")
        )

        if model_type == "ARIMA":
            col_p, col_d, col_q = st.columns(3)
            p = col_p.number_input("p (AR order)", 0, 10, 5)
            d = col_d.number_input("d (I order)", 0, 2, 1)
            q = col_q.number_input("q (MA order)", 0, 10, 2)
            sp = sd = sq = s_period = None
        else:
            col_p, col_d, col_q = st.columns(3)
            p = col_p.number_input("p (AR order)", 0, 10, 1)
            d = col_d.number_input("d (I order)", 0, 2, 1)
            q = col_q.number_input("q (MA order)", 0, 10, 1)

            col_sp, col_sd, col_sq, col_s = st.columns(4)
            sp = col_sp.number_input("P (seasonal AR)", 0, 10, 1)
            sd = col_sd.number_input("D (seasonal I)", 0, 2, 1)
            sq = col_sq.number_input("Q (seasonal MA)", 0, 10, 1)
            s_period = col_s.number_input("Seasonal Period (m)", 1, 365, 7)

        horizon = st.slider("Forecast Horizon (days)", 7, 90, 30)

        if st.button("Run Forecast"):
            try:
                if model_type == "ARIMA":
                    model_ts = ARIMA(ts, order=(p, d, q))
                else:
                    model_ts = SARIMAX(
                        ts, order=(p, d, q), seasonal_order=(sp, sd, sq, s_period),
                        enforce_stationarity=False, enforce_invertibility=False
                    )

                model_fit = model_ts.fit()
                forecast = model_fit.forecast(horizon)
                future_dates = pd.date_range(ts.index.max() + pd.Timedelta(days=1), periods=horizon)

                forecast_df = pd.DataFrame({"date": future_dates, "forecast": forecast.values})

                fig = px.line(daily_demand, x="date", y="units_sold", title=f"{horizon}-Day Forecast")
                fig.add_scatter(x=forecast_df["date"], y=forecast_df["forecast"], mode="lines", name="Forecast")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Model failed: {e}")

# =====================================================
# ================= PRICING INSIGHTS =================
# =====================================================

elif page == "Pricing & Revenue Insights":

    st.header("Pricing and Revenue Insights")

    fig = px.scatter(filtered_df, x="final_price", y="units_sold", color="season", title="Price vs Units Sold by Season")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(filtered_df, x="discount_pct", y="units_sold", color="day_of_week", title="Discount vs Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.line(filtered_df.groupby(["date", "season"])["revenue"].sum().reset_index(),
                  x="date", y="revenue", color="season", title="Revenue Over Time by Season")
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(filtered_df, x="inventory_level", y="units_sold", title="Inventory vs Units Sold")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Business Interpretation")
    st.markdown("""
- Higher prices reduce demand, indicating price sensitivity.  
- Discounts boost sales, especially on certain weekdays.  
- Winter and festive seasons perform best in terms of demand and revenue.  
- Inventory constraints can limit achievable revenue.  
- Competitor pricing has a strong influence on sales volume.  
""")
