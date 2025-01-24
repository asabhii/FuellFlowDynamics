import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load the Dataset
data_url = "India_Statewise_Fuel_Data_Updated_1950_2050 final.csv"
data = pd.read_csv(data_url)

# Streamlit Dashboard Layout
st.title("FuelFlow Dynamics Dashboard")
st.subheader("Explore Fuel Consumption Patterns")

# User Inputs
states = sorted(data['State'].unique())  # Ensure 'State' exists in your dataset
state = st.selectbox("Select State:", states)

# Filter dataset by selected state
state_data = data[data['State'] == state]

# Select Year Range
year_min, year_max = st.slider(
    "Select Year Range:",
    min_value=int(state_data['Year'].min()),
    max_value=int(state_data['Year'].max()),
    value=(1950, 2025)
)

# Filter dataset by selected year range
year_data = state_data[(state_data['Year'] >= year_min) & (state_data['Year'] <= year_max)]

# Select Vehicle Type
vehicle_types = sorted(year_data['Vehicle_Type'].unique())  # Ensure 'Vehicle_Type' exists in your dataset
vehicle_type = st.selectbox("Select Vehicle Type:", vehicle_types)

# Filter dataset by selected vehicle type
final_data = year_data[year_data['Vehicle_Type'] == vehicle_type]

# Display Remaining Data and Charts
st.subheader("Filtered Data")
if not final_data.empty:
    st.write(final_data)
    
    # Chart: Fuel Consumption by Year
    st.subheader("Fuel Consumption Trends")
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if year >= 2025 else 'blue' for year in final_data['Year']]
    sns.barplot(data=final_data, x='Year', y='Fuel_Consumption (Million Liters)', ax=ax, palette=colors)
    ax.set_title(f"Fuel Consumption in {state} ({vehicle_type})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Fuel Consumption (Million Liters)")
    st.pyplot(fig)

    # Additional Chart: Fuel Price Trends
    st.subheader("Fuel Price Trends")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=final_data, x='Year', y='Fuel_Price (INR per Liter)', marker='o', ax=ax2, 
        color="blue", label="1950-2024"
    )
    sns.scatterplot(
        data=final_data[final_data['Year'] >= 2025], x='Year', y='Fuel_Price (INR per Liter)', 
        color="red", s=100, label="2025-2050"
    )
    ax2.set_title(f"Fuel Price Trends in {state} ({vehicle_type})")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Fuel Price (INR per Liter)")
    ax2.legend()
    st.pyplot(fig2)

    # Additional Chart: Population Trends
    st.subheader("Population Trends")
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=final_data, x='Year', y='Population', marker='o', ax=ax3, 
        color="blue", label="1950-2024"
    )
    sns.scatterplot(
        data=final_data[final_data['Year'] >= 2025], x='Year', y='Population', 
        color="red", s=100, label="2025-2050"
    )
    ax3.set_title(f"Population Trends in {state} ({vehicle_type})")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Population")
    ax3.legend()
    st.pyplot(fig3)
else:
    st.write("No data available for the selected filters.")

# Summary Insights
st.subheader("Summary Insights")
if not final_data.empty:
    total_consumption = final_data['Fuel_Consumption (Million Liters)'].sum()
    avg_price = final_data['Fuel_Price (INR per Liter)'].mean()
    population = final_data['Population'].mean()

    st.write(f"Total Fuel Consumption: {total_consumption:.2f} Million Liters")
    st.write(f"Average Fuel Price: {avg_price:.2f} INR per Liter")
    st.write(f"Average Population: {population:.0f}")
else:
    st.write("No insights available for the selected filters.")

