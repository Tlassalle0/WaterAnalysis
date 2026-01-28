import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#----------------------------------------
# Page config
st.set_page_config(
    page_title="Water Intake Analysis",
    page_icon="💧",
    layout="wide",
)
#----------------------------------------


#----------------------------------------
# Load data
df=pd.read_csv('C:/Users/MV_pe/OneDrive/Documents/Ynov/B3_(2025-2026)/Analyse et Exploration de Données/Projet/WaterAnalysis/data/2_cleaned/water_intake_cleaned.csv')

st.title("Water Intake Analysis 📊")



#----------------------------------------







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
tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs(["Age", "Poids", "Genre", "Activité Physique", "Niveau d'Hydratation", "Apport quotidien en Eau"])






with tab1:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Age"].dropna())
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Age")
        mean_age = df["Age"].mean()
        median_age = df["Age"].median()
        ax.axvline(mean_age,color='red', linestyle='dashed', linewidth=1 , label=f"Mean age: {mean_age:.1f}")
        ax.axvline(median_age,color='green', linestyle='dashed', linewidth=1 , label=f"Median age: {median_age:.1f}")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.write("""
        **Observations:**
        - L'âge moyen des individus dans le dataset est d'environ 43 ans.
        - L'âge médian est également proche de 43 ans, indiquant une distribution relativement symétrique.
        - Cette symétrie suggère qu'il n'y a pas de biais significatif vers les jeunes ou les personnes âgées dans l'échantillon ou dans le cas présent que le datatset à été généré.
        """)




with tab2:
    col1, col2 = st.columns([1, 1])  
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Weight"].dropna())
        ax.set_xlabel("Weight")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Weight")
        mean_weight = df["Weight"].mean()
        median_weight = df["Weight"].median()
        ax.axvline(mean_weight,color='red', linestyle='dashed', linewidth=1 , label=f"Mean weight: {mean_weight:.1f}")
        ax.axvline(median_weight,color='green', linestyle='dashed', linewidth=1 , label=f"Median weight: {median_weight:.1f}")
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.write("""
        **Observations:**
        - Le poids moyen des individus dans le dataset est d'environ 75 kg.
        - Le poids médian est également proche de 77 kg, indiquant une distribution relativement symétrique.
        - Cette symétrie suggère qu'il n'y a pas de biais significatif vers les individus plus légers ou plus lourds dans l'échantillon ou dans le cas présent que le datatset à été généré.
        """)
    

with tab3:
    col1, col2 = st.columns([1, 1])  
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Gender"].dropna())
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Gender")
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - La répartition des genres dans le dataset est relativement équilibrée entre les hommes et les femmes.
        - Cela suggère que l'échantillon est soit très équilibré, soit que le dataset a été généré pour inclure un nombre égal d'hommes et de femmes.
        """)



with tab4:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Physical Activity Level"].dropna())
        ax.set_xlabel("Physical Activity Level")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Physical Activity Level")
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On retrouve une répartition encore parfaite entre les différents niveaux d'activité physique.
        - Cela suggère que le dataset a été généré pour inclure un nombre égal d'individus dans chaque catégorie d'activité physique.
        """)



with tab5:
    col1, col2 = st.columns([1, 1])  # graph takes 1/3 of width
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Hydration Level"].dropna())
        ax.set_xlabel("Hydration Level")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Hydration Level")
        
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - Une proportion significative des individus dans le dataset sont classés comme ayant un niveau d'hydratation "Good".
        """)

with tab6:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.histplot(
            data=df,
            x="Daily Water Intake",
            bins=20,
            kde=True,
            color="blue",
            ax=ax
        )
        ax.set_xlabel("Daily Water Intake (liters)")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Daily Water Intake")
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - une quantité très élevée de personne consomme seulement 1,5 litres d'eau par jour.
        - La consommation quotidienne d'eau des individus dans le dataset suit une distribution normale.
        - La majorité des individus consomment entre 2 et 3,5 litres d'eau par jour, avec une moyenne autour de 3 litres.
        - Quelques individus consomment des quantités très élevées d'eau, ce qui pourrait indiquer des besoins spécifiques ou des comportements particuliers comme le poids ou la température.
        """)


df_good = df[df["Hydration Level"] == "Poor"]

weight_bins = [0, 60, 80, 100, float("inf")]
weight_labels = ["<60 kg", "60–80 kg", "80–100 kg", ">100 kg"]

df_good["Weight Group"] = pd.cut(
    df_good["Weight"],
    bins=weight_bins,
    labels=weight_labels
)

grid_mean = (
    df_good.groupby(["Weight Group", "Weather"])["Daily Water Intake"]
      .agg(["mean"])
      .round(2)
)

grid_mean_table = grid_mean.unstack(level="Weather")

grid_mean_table = grid_mean_table.reindex(
    columns=["Cold", "Normal", "Hot"],
    level=1
)

grid_med = (
    df_good.groupby(["Weight Group", "Weather"])["Daily Water Intake"]
      .agg(["median"])
      .round(2)
)
grid_med_table = grid_med.unstack(level="Weather")
grid_med_table = grid_med_table.reindex(
    columns=["Cold", "Normal", "Hot"],
    level=1
)




st.subheader("MEAN & MEDIAN Daily Water Intake")
tab1, tab2= st.tabs(["Mean", "Median"])

with tab1:
    st.dataframe(
        grid_mean_table,
        use_container_width=True
    )
with tab2:
    st.dataframe(
        grid_med_table,
        use_container_width=True
    )


bins = [44.99, 61, 77, 93, 109.01]
labels = ['44.99-61 kg', '62-77 kg', '78-93 kg', '94-109.01 kg']
df_grpWeight = df.copy()
df_grpWeight['Weight Group'] = pd.cut(df_grpWeight['Weight'], bins=bins, labels=labels,include_lowest=True)
palette_weight = {
    '44.99-61 kg': 'lightgreen',
    '62-77 kg': 'orange',
    '78-93 kg': 'lightcoral',
    '94-109.01 kg': 'purple'
}



df_grpWeight['Weather'] = pd.Categorical(
    df['Weather'],
    categories=['Cold', 'Normal', 'Hot'],
    ordered=True
)

sns.boxplot(data=df_grpWeight, x="Weather", y="Daily Water Intake", hue="Weight Group", palette=palette_weight)