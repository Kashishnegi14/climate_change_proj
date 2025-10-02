import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os

# Set page configuration for a wider layout
st.set_page_config(layout="wide", page_title="Climate Change EDA Dashboard", page_icon="🌍")

# --- Data Loading and Preprocessing ---
@st.cache_data # Cache data loading to improve performance
def load_data():
    try:
        df = pd.read_csv('climate_change_dataset.csv')
        # Rename columns for easier access and cleaner display
        df.columns = ['Year', 'Country', 'Avg_Temperature_°C', 'CO2_Emissions_Tons/Capita',
                      'Sea_Level_Rise_mm', 'Rainfall_mm', 'Population', 'Renewable_Energy_%',
                      'Extreme_Weather_Events', 'Forest_Area_%']
        
        # Handle missing values - a simple imputation for numerical columns
        for col in df.select_dtypes(include=np.number).columns:
            if col != 'Year': # Don't fill Year
                df[col] = df[col].fillna(df[col].median()) # Using median for robustness to outliers

        # Calculate country-level aggregates for summary and top N charts
        country_stats = df.groupby('Country').agg(
            Avg_Temp=('Avg_Temperature_°C', 'mean'),
            Avg_CO2_Emissions=('CO2_Emissions_Tons/Capita', 'mean'),
            Avg_Sea_Level_Rise=('Sea_Level_Rise_mm', 'mean'),
            Avg_Rainfall=('Rainfall_mm', 'mean'),
            Avg_Renewable_Energy=('Renewable_Energy_%', 'mean'),
            Total_Extreme_Weather_Events=('Extreme_Weather_Events', 'sum'),
            Avg_Forest_Area=('Forest_Area_%', 'mean'),
            Avg_Population=('Population', 'mean')
        ).reset_index()
        
        # Sort for top N later
        country_stats = country_stats.sort_values(by='Avg_CO2_Emissions', ascending=False)

        return df, country_stats
    except FileNotFoundError:
        st.error("Error: `climate_change_dataset.csv` not found. Please upload the file or ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        st.stop()

df, country_stats = load_data()

# --- Key Insights and Policy Recommendations ---
insights = [
    "A strong positive correlation exists between CO2 emissions and average temperature rise, highlighting anthropogenic impact.",
    "Countries with higher renewable energy adoption percentages tend to show lower or stabilized CO2 emission growth.",
    "Sea level rise is accelerating, particularly in recent decades, closely tracking global temperature increases.",
    "Forest cover plays a crucial role; regions with greater forest area exhibit more stable temperatures and potentially fewer extreme weather events.",
    "Population density, combined with climate factors, increases vulnerability to extreme weather events, necessitating targeted urban resilience strategies."
]

policies = [
    "**Global Carbon Pricing:** Implement a standardized global carbon pricing mechanism to incentivize emission reductions across all sectors.",
    "**Renewable Energy Subsidies:** Increase subsidies and investment in renewable energy technologies and infrastructure, especially in developing economies.",
    "**Reforestation & Conservation:** Fund large-scale reforestation and afforestation projects, coupled with strict anti-deforestation policies.",
    "**Coastal Adaptation:** Invest in climate-resilient coastal infrastructure and early warning systems for communities vulnerable to sea level rise.",
    "**Urban Resilience Programs:** Develop and fund urban planning initiatives focused on adapting cities to extreme weather events, particularly in high-density areas."
]

# --- Dashboard Layout and Content ---
st.title("🌍 Global Climate Change: An Exploratory Data Analysis")
st.markdown("---")
st.markdown("""
    Welcome to the Deep Data Hackathon 2.0 - Round 1 Climate Change Dashboard! 
    This interactive tool allows policymakers and researchers to explore critical climate indicators, 
    identify key trends, and inform strategic decisions for a sustainable future.
""")
st.markdown("---")

# --- Sidebar Controls ---
st.sidebar.header("Dashboard Controls")
country_list = ["All"] + sorted(df['Country'].unique().tolist())
selected_country = st.sidebar.selectbox("Select a Country", country_list)

metric_options = ['Avg_Temperature_°C', 'CO2_Emissions_Tons/Capita',
                  'Sea_Level_Rise_mm', 'Rainfall_mm', 'Population',
                  'Renewable_Energy_%', 'Extreme_Weather_Events', 'Forest_Area_%']
default_metrics = ['Avg_Temperature_°C', 'CO2_Emissions_Tons/Capita', 'Sea_Level_Rise_mm']
selected_metrics_time_series = st.sidebar.multiselect(
    "Select Metrics for Yearly Trends",
    metric_options,
    default=default_metrics
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dynamic Scatter Plot Options")
scatter_x_metric = st.sidebar.selectbox("Select X-axis Metric", metric_options, index=1) # Default CO2 Emissions
scatter_y_metric = st.sidebar.selectbox("Select Y-axis Metric", metric_options, index=0) # Default Avg Temperature


# --- Filter Dataset based on Sidebar Selection ---
if selected_country != "All":
    df_filtered = df[df['Country'] == selected_country].copy()
else:
    df_filtered = df.copy()

# --- Main Content Area ---

# Row 1: Yearly Trends
st.subheader("📈 Yearly Trends of Climate Indicators")
st.markdown("Explore how various climate metrics have changed over time globally or for a specific country.")
if not selected_metrics_time_series:
    st.info("Please select at least one metric from the sidebar to display yearly trends.")
else:
    # Group by year and take the mean for trends
    yearly_data = df_filtered.groupby('Year')[selected_metrics_time_series].mean().reset_index()
    
    fig_line = go.Figure()
    for metric in selected_metrics_time_series:
        fig_line.add_trace(go.Scatter(x=yearly_data['Year'], y=yearly_data[metric], mode='lines+markers', name=metric))
    
    fig_line.update_layout(
        title=f"Yearly Trends for {selected_country}" if selected_country != "All" else "Global Yearly Trends",
        xaxis_title="Year",
        yaxis_title="Value",
        hovermode="x unified",
        legend_title="Metric",
        height=450,
        template="plotly_white"
    )
    st.plotly_chart(fig_line, use_container_width=True)
st.markdown("---")

# Row 2: Correlation Heatmap and Dynamic Scatter Plot
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔗 Correlation Heatmap")
    st.markdown("Understand the relationships between selected climate indicators.")
    
    # Ensure there are at least 2 selected metrics for correlation
    if len(selected_metrics_time_series) < 2:
        st.info("Select at least two metrics for the heatmap from the 'Yearly Trends' sidebar options.")
    else:
        corr_matrix = df_filtered[selected_metrics_time_series].corr()
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)
        plt.title(f"Correlation Matrix for {selected_country}" if selected_country != "All" else "Global Correlation Matrix")
        st.pyplot(fig_corr)
        # Save plot functionality
        if st.button("Download Heatmap"):
            fig_corr.savefig("heatmap.png", bbox_inches='tight')
            with open("heatmap.png", "rb") as file:
                st.download_button(
                    label="Click to Download",
                    data=file,
                    file_name="correlation_heatmap.png",
                    mime="image/png"
                )

with col2:
    st.subheader("🔍 Dynamic Scatter Plot")
    st.markdown("Visualize the relationship between any two chosen metrics.")
    
    # Check if selected metrics exist in the filtered dataframe
    if scatter_x_metric in df_filtered.columns and scatter_y_metric in df_filtered.columns:
        fig_scatter = px.scatter(
            df_filtered,
            x=scatter_x_metric,
            y=scatter_y_metric,
            color='Year' if selected_country == "All" else None, # Color by year only for global view
            hover_name="Country",
            title=f"{scatter_x_metric} vs {scatter_y_metric} ({selected_country})" if selected_country != "All" else f"{scatter_x_metric} vs {scatter_y_metric} (Global)",
            template="plotly_white",
            labels={scatter_x_metric: scatter_x_metric.replace('_', ' ').replace('°C', '°C').replace('%', '%').replace('mm', 'mm').replace('Tons/Capita', 'Tons/Capita'),
                    scatter_y_metric: scatter_y_metric.replace('_', ' ').replace('°C', '°C').replace('%', '%').replace('mm', 'mm').replace('Tons/Capita', 'Tons/Capita')}
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)
        # Save plot functionality
        if st.button("Download Scatter Plot"):
            fig_scatter.write_image("scatterplot.png")
            with open("scatterplot.png", "rb") as file:
                st.download_button(
                    label="Click to Download",
                    data=file,
                    file_name="dynamic_scatter_plot.png",
                    mime="image/png"
                )
    else:
        st.warning("Please ensure both selected metrics are available in the dataset.")
st.markdown("---")

# Row 3: Country-Level Summaries and Top Emitters
st.subheader("🌐 Country-Level Summaries and Rankings")
st.markdown("A quick overview of key climate indicators for each country and top CO2 emitters.")

col3, col4 = st.columns([2, 1])

with col3:
    st.markdown("#### All Countries Summary Table")
    st.dataframe(country_stats.style.format({
        'Avg_Temp': "{:.2f}°C",
        'Avg_CO2_Emissions': "{:.2f}",
        'Avg_Sea_Level_Rise': "{:.2f}mm",
        'Avg_Rainfall': "{:.0f}mm",
        'Avg_Renewable_Energy': "{:.1f}%",
        'Total_Extreme_Weather_Events': "{:.0f}",
        'Avg_Forest_Area': "{:.1f}%",
        'Avg_Population': "{:,.0f}"
    }), use_container_width=True, height=400)

with col4:
    st.markdown("#### Top 10 Countries by Average CO2 Emissions")
    top_10_co2 = country_stats.nlargest(10, 'Avg_CO2_Emissions')
    fig_top_co2 = px.bar(
        top_10_co2,
        x='Avg_CO2_Emissions',
        y='Country',
        orientation='h',
        title='Top 10 Countries by Avg. CO2 Emissions (Tons/Capita)',
        labels={'Avg_CO2_Emissions': 'Avg. CO2 Emissions (Tons/Capita)', 'Country': 'Country'},
        template="plotly_white",
        color='Avg_CO2_Emissions',
        color_continuous_scale=px.colors.sequential.Reds
    )
    fig_top_co2.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_top_co2, use_container_width=True)
st.markdown("---")

# Row 4: Key Insights and Policy Recommendations
st.subheader("💡 Key Insights & 📌 Policy Recommendations")
st.markdown("Based on our analysis, here are crucial findings and actionable policy steps.")

col5, col6 = st.columns(2)

with col5:
    st.markdown("#### Key Insights")
    for i, insight in enumerate(insights):
        st.info(f"**Insight {i+1}:** {insight}")

with col6:
    st.markdown("#### Policy Recommendations")
    for i, policy in enumerate(policies):
        st.success(f"**Policy {i+1}:** {policy}")
st.markdown("---")

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 12px;
    }
    </style>
    <div class="footer">
        Developed for Deep Data Hackathon 2.0 - Round 1 | Data Source: Climate Change Dataset
    </div>
    """, unsafe_allow_html=True)

