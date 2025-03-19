import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np

# Load the dataset
file_path = "data/Competition_Data.csv"
df = pd.read_csv(file_path)

# Rename columns to maintain consistency
df.rename(columns={
    "Fiscal_Week_ID": "Fiscal_Week_Id",
    "Store_ID": "Store_Id",
    "Item_ID": "Item_Id"
}, inplace=True)

# Convert Fiscal_Week_Id to datetime format (if applicable)
df['Fiscal_Week_Id'] = pd.to_datetime(df['Fiscal_Week_Id'], errors='coerce')

# Streamlit App Setup
st.title("📊 Price Optimization Analysis Dashboard")
st.markdown("### An interactive tool for analyzing pricing strategies and competition.")

# Sidebar Filters
store_filter = st.sidebar.selectbox("Select Store:", options=["All"] + list(df['Store_Id'].unique()))
item_filter = st.sidebar.selectbox("Select Item:", options=["All"] + list(df['Item_Id'].unique()))

# Filter Data based on selection
filtered_df = df.copy()
if store_filter != "All":
    filtered_df = filtered_df[filtered_df['Store_Id'] == store_filter]
if item_filter != "All":
    filtered_df = filtered_df[filtered_df['Item_Id'] == item_filter]

# Price Elasticity of Demand
st.subheader("Price Elasticity of Demand")
fig = px.scatter(filtered_df, x='Price', y='Item_Quantity', title='Price vs. Item Quantity', color='Price', hover_data=['Sales_Amount'])
st.plotly_chart(fig)

# Revenue Heatmap
st.subheader("Revenue Heatmap")
pivot_table = filtered_df.pivot_table(values='Sales_Amount', index='Store_Id', columns='Fiscal_Week_Id', aggfunc='sum')
fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(pivot_table, cmap="coolwarm", annot=False, ax=ax)
st.pyplot(fig)

# Sales and Competition Price Comparison
st.subheader("Sales vs. Competition Price")
filtered_df['Competition_Price'].fillna(filtered_df['Competition_Price'].median(), inplace=True)
fig = px.line(filtered_df, x='Fiscal_Week_Id', y=['Sales_Amount', 'Competition_Price'], title='Sales vs. Competitor Price', color_discrete_map={'Sales_Amount':'blue', 'Competition_Price':'red'}, hover_data=['Price'])
st.plotly_chart(fig)

# Profitability Analysis
filtered_df['Profit'] = filtered_df['Sales_Amount'] - (filtered_df['Item_Quantity'] * filtered_df['Price'])
st.subheader("Profitability Over Time")
fig = px.line(filtered_df, x='Fiscal_Week_Id', y='Profit', title='Profit Over Time', color_discrete_sequence=['green'], hover_data=['Sales_Amount'])
st.plotly_chart(fig)

# Advanced Forecasted Sales (Polynomial Regression Prediction)
st.subheader("Sales Prediction")
filtered_df = filtered_df.dropna(subset=['Fiscal_Week_Id'])
filtered_df['Week_Num'] = filtered_df['Fiscal_Week_Id'].dt.strftime('%Y%U')
filtered_df['Week_Num'] = pd.to_numeric(filtered_df['Week_Num'], errors='coerce')
filtered_df = filtered_df.dropna(subset=['Week_Num'])
filtered_df['Week_Num'] = filtered_df['Week_Num'].astype(int)

X = filtered_df[['Week_Num']]
y = filtered_df['Sales_Amount']

poly_model = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
poly_model.fit(X, y)
filtered_df['Predicted_Sales'] = poly_model.predict(X)

fig = px.line(filtered_df, x='Fiscal_Week_Id', y=['Sales_Amount', 'Predicted_Sales'], title='Actual vs. Predicted Sales', color_discrete_map={'Sales_Amount':'blue', 'Predicted_Sales':'orange'}, hover_data=['Price'])
st.plotly_chart(fig)

# Enhanced Visualizations
st.subheader("Price vs. Sales Amount")
fig = px.scatter(filtered_df, x='Price', y='Sales_Amount', title='Price vs. Sales Amount', color='Sales_Amount', hover_data=['Item_Quantity'])
st.plotly_chart(fig)

st.subheader("Our Price vs. Competitor Price Distribution")
fig = px.histogram(filtered_df, x=['Price', 'Competition_Price'], title='Our Price vs. Competitor Price Distribution', barmode='overlay', histnorm='percent', color_discrete_sequence=['blue', 'red'])
st.plotly_chart(fig)

st.subheader("Sales Trend Over Time")
fig = px.line(filtered_df, x='Fiscal_Week_Id', y='Sales_Amount', title='Sales Trend Over Time', color_discrete_sequence=['purple'], hover_data=['Price'])
st.plotly_chart(fig)
