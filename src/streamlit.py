import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Page config
st.set_page_config(
    page_title="Water Intake Analysis",
    page_icon="💧",
    layout="wide",

)
# Load data
df=pd.read_csv('C:/Users/MV_pe/OneDrive/Documents/Ynov/B3_(2025-2026)/Analyse et Exploration de Données/Projet/WaterAnalysis/data/2-interim/Daily_Water_Intake_Cleaned.csv')




st.title("Water Intake Analysis 📊")
st.write("Hello Worlds!")



# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Accueil", "Analyse", "Visualisation"])


st.subheader("Daily Water Intake by Gender")









fig, ax = plt.subplots(figsize=(8, 6))

palette = {
    'Male': 'skyblue',
    'Female': 'lightpink'
}

# Boxplot
sns.boxplot(
    data=df,
    x="Gender",
    y="Daily Water Intake (liters)",
    palette=palette,
    ax=ax
)

# Labels
ax.set_title("Daily Water Intake by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Daily Water Intake (liters)")

# Display in Streamlit
st.pyplot(fig, use_container_width=True)
