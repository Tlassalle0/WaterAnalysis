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
df=pd.read_csv('C:/Users/MV_pe/OneDrive/Documents/Ynov/B3_(2025-2026)/Analyse et Exploration de Données/Projet/WaterAnalysis/data/2_cleaned/water_intake_cleaned.csv')




st.title("Water Intake Analysis 📊")
st.write("Hello Worlds!")



# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller à", ["Overview", "Analyse", "Visualisation"])


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
    y="Daily Water Intake",
    palette=palette,
    ax=ax
)

# Labels
ax.set_title("Daily Water Intake by Gender")
ax.set_xlabel("Gender")
ax.set_ylabel("Daily Water Intake (liters)")

# Display in Streamlit
st.pyplot(fig, use_container_width=True)



# Tabs
st.subheader("Overview Tabs")
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Age", "Poids", "Genre", "Activité Physique", "Niveau d'Hydratation"])

with tab1:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(df["Age"].dropna())
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Age")
    mean_age = df["Age"].mean()
    ax.axvline(mean_age,color='red', linestyle='dashed', linewidth=1 , label=f"Mean age: {mean_age:.1f}")
    ax.legend()
    st.pyplot(fig,width=700)




with tab2:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(df["Weight"].dropna())
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Weight")
    mean_weight = df["Weight"].mean()
    ax.axvline(mean_weight,color='red', linestyle='dashed', linewidth=1 , label=f"Mean weight: {mean_weight:.1f}")
    ax.legend()
    st.pyplot(fig,width=700)

with tab3:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(df["Gender"].dropna())
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Gender")
    

    st.pyplot(fig,width=700)

with tab4:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(df["Physical Activity Level"].dropna())
    ax.set_xlabel("Physical Activity Level")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Physical Activity Level")

    st.pyplot(fig,width=700)

with tab5:
    fig, ax = plt.subplots(figsize=(4,3))

    ax.hist(df["Hydration Level"].dropna())
    ax.set_xlabel("Hydration Level")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Hydration Level")

    st.pyplot(fig,width=700)