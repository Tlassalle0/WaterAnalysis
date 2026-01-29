import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Page config
st.set_page_config(
    page_title="Analyse de l'hydratation dans le monde",
    page_icon="💧",
    layout="wide",

)
# Load data
df = pd.read_csv('data/2_cleaned/water_intake_cleaned.csv')
df_grpWeight=df.copy()
palette_weight = {
    '<60 kg': 'lightgreen',
    '60–80 kg': 'orange',
    '80–100 kg': 'lightcoral',
    '>100 kg': 'purple'
}

weight_bins = [0, 60, 80, 100, float("inf")]
weight_labels = ["<60 kg", "60–80 kg", "80–100 kg", ">100 kg"]

df_grpWeight["Weight Group"] = pd.cut(
    df_grpWeight["Weight"],
    bins=weight_bins,
    labels=weight_labels
)

age_bins = [18, 30, 40, 50, 60, 70]
age_labels = ["18–29", "30–39", "40–49", "50–59", "60–70"]

df["Age Group"] = pd.cut(
    df["Age"],
    bins=age_bins,
    labels=age_labels,
    include_lowest=True
)




st.title("Analyse de l'hydratation dans le monde 📊")
st.write("""Ce dashboard présente une analyse des données relatives à la consommation d'eau quotidienne des individus, en fonction de divers facteurs tels que l'âge, le poids, le genre, le niveau d'activité physique, le niveau d'hydratation et les conditions météorologiques.""")#----------------------------------------




# Tabs
st.subheader("Data originel")
tab1, tab2, tab3, tab4, tab5 ,tab6= st.tabs(["Age", "Poids", "Genre", "Activité Physique", "Niveau d'Hydratation", "Apport quotidien en Eau"])






with tab1:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Age"].dropna())
        ax.set_xlabel("Age")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition de l'âge")
        mean_age = df["Age"].mean()
        median_age = df["Age"].median()
        ax.axvline(mean_age,color='red', linestyle='dashed', linewidth=1 , label=f"Âge moyen: {mean_age:.1f}")
        ax.axvline(median_age,color='green', linestyle='dashed', linewidth=1 , label=f"Âge médian: {median_age:.1f}")
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
        ax.set_xlabel("Poids")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition du Poids")
        mean_weight = df["Weight"].mean()
        median_weight = df["Weight"].median()
        ax.axvline(mean_weight,color='red', linestyle='dashed', linewidth=1 , label=f"Poids moyen: {mean_weight:.1f}")
        ax.axvline(median_weight,color='green', linestyle='dashed', linewidth=1 , label=f"Poids médian: {median_weight:.1f}")
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
        ax.set_xlabel("Genre")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition des genres")
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
        ax.set_xlabel("Activité Physique")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition de l'activité physique")
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On retrouve une répartition encore parfaite entre les différents niveaux d'activité physique.
        - Cela suggère que le dataset a été généré pour inclure un nombre égal d'individus dans chaque catégorie d'activité physique.
        """)



with tab5:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        fig, ax = plt.subplots(figsize=(4,3))

        ax.hist(df["Hydration Level"].dropna())
        ax.set_xlabel("Niveau d'hydratation")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition du niveau d'hydratation")
        
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
        ax.set_xlabel("Consommation d'eau")
        ax.set_ylabel("Nombre")
        ax.set_title("Répartition de la consommation d'eau quotidienne")
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - une quantité très élevée de personne consomme seulement 1,5 litres d'eau par jour.
        - La consommation quotidienne d'eau des individus dans le dataset suit une distribution normale.
        - La majorité des individus consomment entre 2 et 3,5 litres d'eau par jour, avec une moyenne autour de 3 litres.
        - Quelques individus consomment des quantités très élevées d'eau, ce qui pourrait indiquer des besoins spécifiques ou des comportements particuliers comme le poids ou la température.
        """)

#----------------------------------------
# Mean & Median Poor Daily Water Intake Table


weight_bins = [0, 60, 80, 100, float("inf")]
weight_labels = ["<60 kg", "60–80 kg", "80–100 kg", ">100 kg"]

df["Weight Group"] = pd.cut(
    df["Weight"],
    bins=weight_bins,
    labels=weight_labels
)


df_good = df[df["Hydration Level"] == "Good"]
df_bad = df[df["Hydration Level"] == "Poor"]


def make_tables(dataframe):
    grid_mean = (
    dataframe.groupby(["Weight Group", "Weather"])["Daily Water Intake"]
      .agg(["mean"])
      .round(2)
      )
    grid_mean_table = grid_mean.unstack(level="Weather")

    grid_mean_table = grid_mean_table.reindex(
        columns=["Cold", "Normal", "Hot"],
        level=1
    )

    grid_med = (
        dataframe.groupby(["Weight Group", "Weather"])["Daily Water Intake"]
        .agg(["median"])
        .round(2)
    )
    grid_med_table = grid_med.unstack(level="Weather")
    grid_med_table = grid_med_table.reindex(
        columns=["Cold", "Normal", "Hot"],
        level=1
    )

    return grid_mean_table, grid_med_table








good_tables = make_tables(df_good)
bad_tables = make_tables(df_bad)

st.subheader("Moyennes & Médianes de la consommation d'eau par niveau d'hydratation")
tab1, tab2= st.tabs(["Moyennes", "Médianes"])

with tab1:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        st.markdown("Suffisant")
        st.dataframe(
            good_tables[0],
            use_container_width=True
        )
    with col2:
        st.markdown("Insuffisant")
        st.dataframe(
            bad_tables[0],
            use_container_width=True
        )
    
    mean_diff = (good_tables[0] - bad_tables[0]).stack().mean().round(2).item()
    st.markdown(f"La moyenne de la différence de consommation d'eau quotidienne entre les niveaux d'hydratation bon et mauvais est de : **{mean_diff}** litres.")
with tab2:
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("Suffisant")
        st.dataframe(
            good_tables[1],
            use_container_width=True
        )
    with col2:
        st.markdown("Insuffisant")
        st.dataframe(
            bad_tables[1],
            use_container_width=True
        )
    median_diff = (good_tables[1] - bad_tables[1]).stack().mean().round(2).item()
    st.markdown(f"La médiane de la différence de consommation d'eau quotidienne entre les niveaux d'hydratation bon et mauvais est de : **{median_diff}** litres.")

st.subheader("Analyses")
tab1, tab2, tab3 ,tab4,tab5,tab6,tab7= st.tabs(["Genre", "Activité Physique", "Climat" ,"Age", "Poids","Poids & Climat","Age & Climat"])

with tab1:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        
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
        ax.set_title("Consommation par genre")
        ax.set_xlabel("Genre")
        ax.set_ylabel("Consommation d'eau quotidienne")

        # Display in Streamlit
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.write("""
        **Observations:**
        - On observe que le genre n'a pas d'impact significatif sur la consommation d'eau quotidienne.
        - La similarité execive entre les deux boxplots souligne encore la possible .
        """)
    
with tab2:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        df['Physical Activity Level'] = pd.Categorical(
            df['Physical Activity Level'],
            categories=['Low', 'Moderate', 'High'],
            ordered=True
        )

        # Compute mean water intake
        activity_water = (
            df.groupby("Physical Activity Level")["Daily Water Intake"]
            .mean()
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(5, 4))

        ax.bar(
            activity_water.index,
            activity_water.values
        )

        ax.set_title("Consommation quotidienne d'eau par niveau d'activité physique")
        ax.set_xlabel("Acitivité physique")
        ax.set_ylabel("Consommation d'eau quotidienne")

        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - Il y a une tendance claire indiquant que les individus avec des niveaux d'activité physique plus élevés consomment davantage d'eau quotidiennement.
        - Cela est cohérent avec les attentes, car une activité physique accrue entraîne une perte de fluides corporels, nécessitant une hydratation supplémentaire.
        """)


with tab3:
    col1, col2 = st.columns([1, 1]) 
    with col1:
        # Ensure ordered category
        df['Weather'] = pd.Categorical(
            df['Weather'],
            categories=['Cold', 'Normal', 'Hot'],
            ordered=True
        )

        # Count occurrences
        counts = (
            df.groupby(["Weather", "Hydration Level"])
            .size()
            .unstack(fill_value=0)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))

        counts.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_xlabel("Climat")
        ax.set_ylabel("Nombre")
        ax.set_title("Qualité de l'hydratation selon le climat")
        ax.legend(title="Niveau d'hydratation")

        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On observe que les conditions météorologiques influencent le niveau d'hydratation des individus.
        - Par temps froid, une proportion plus élevée d'individus présente un niveau d'hydratation "Poor", ce qui suggère que la froid peut réduire la sensation de soif et donc la consommation d'eau.""")


with tab4:
    col1, col2 = st.columns([1, 1]) 
    with col1:


        # Group counts
        counts = (
            df.groupby(["Age Group", "Hydration Level"])
            .size()
            .unstack(fill_value=0)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))

        counts.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_xlabel("Groupe d'âge")
        ax.set_ylabel("Nombre")
        ax.set_title("Niveau d'hydratation selon l'âge")
        ax.legend(title="Niveau d'hydratation")

        # Totals per age group
        totals = counts.sum(axis=1)

        # Add percentage labels
        for container, level in zip(ax.containers, counts.columns):
            for rect, total in zip(container, totals):
                height = rect.get_height()
                if height > 0:
                    percentage = height / total * 100
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                    )

        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On observe que les individus plus agés ont tendance à avoir un taux de niveau d'hydratation "Poor" plus élevé.
        - Cela peut s'expliquer par le fait que la sensation de soif diminue avec l'âge, ce qui peut entraîner une consommation d'eau insuffisante chez les personnes âgées.
        """)

with tab5:
    col1, col2 = st.columns([1, 1])
    with col1:


        st.subheader("Hydratation selon le poids")

        # Group counts
        counts = (
            df.groupby(["Weight Group", "Hydration Level"])
            .size()
            .unstack(fill_value=0)
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))

        counts.plot(
            kind="bar",
            stacked=True,
            ax=ax
        )

        ax.set_xlabel("Poids")
        ax.set_ylabel("Nombre")
        ax.set_title("Hydratation selon le poids")
        ax.legend(title="Niveau d'hydratation")

        # Totals per weight group
        totals = counts.sum(axis=1)

        # Add percentage labels
        for container, level in zip(ax.containers, counts.columns):
            for rect, total in zip(container, totals):
                height = rect.get_height()
                if height > 0:
                    percentage = height / total * 100
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2,
                        f"{percentage:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white"
                    )

        plt.tight_layout()

        # Display in Streamlit
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On observe que les individus en dessous de 60 kg ont tendance à avoir un taux de niveau d'hydratation "Good" plus élevé.
        - Cela peut s'expliquer par le fait que les individus avec un poids corporel plus léger ont des besoins en hydratation plus faible.
        """)



with tab6:
    col1, col2 = st.columns([1, 1]) 
    with col1:

        # Ensure ordered category
        df_grpWeight['Weather'] = pd.Categorical(
            df_grpWeight['Weather'],
            categories=['Cold', 'Normal', 'Hot'],
            ordered=True
        )

        # Create figure
        fig, ax = plt.subplots(figsize=(6, 4))

        sns.boxplot(
            data=df_grpWeight,
            x="Weather",
            y="Daily Water Intake",
            hue="Weight Group",
            palette=palette_weight,
            ax=ax
        )

        ax.set_title("Consommation d'eau par poids et par climat")
        ax.set_xlabel("Climat")
        ax.set_ylabel("Consommation d'eau quotidienne")
        ax.legend(title="Poids")

        # Display in Streamlit
        st.pyplot(fig)
    
    with col2:
        st.write("""
        **Observations:**
        - On observe que les individus plus lourds ont tendance à consommer davantage d'eau, ce qui est logique car un poids corporel plus élevé nécessite une hydratation accrue.
        - De plus, la consommation d'eau augmente avec la température, ce qui est cohérent avec le fait que les besoins en hydratation augmentent par temps chaud en raison de la transpiration accrue.
        """)


with tab7:
    col1, col2 = st.columns([1, 1])
    with col1:
            

        counts = (
            df.groupby(["Age Group", "Weather", "Hydration Level"])
            .size()
            .unstack(fill_value=0)
        )
        # Order definitions
        age_groups = counts.index.get_level_values("Age Group").unique()
        weathers = ["Cold", "Normal", "Hot"]

        x = np.arange(len(age_groups))
        width = 0.25

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, weather in enumerate(weathers):
            data = counts.xs(weather, level="Weather")

            good = data.get("Good", 0)
            poor = data.get("Poor", 0)

            # Good hydration (blue)
            ax.bar(
                x + i * width,
                good,
                width,
                color="tab:blue",
                edgecolor="black",
                linewidth=0.8,
                label="Good" if i == 0 else None
            )

            # Poor hydration (orange)
            ax.bar(
                x + i * width,
                poor,
                width,
                bottom=good,
                color="tab:orange",
                edgecolor="black",
                linewidth=0.8,
                label="Poor" if i == 0 else None
            )

        # Axis & labels
        ax.set_xticks(x + width)
        ax.set_xticklabels(age_groups)
        ax.set_xlabel("Groupe d'âge")
        ax.set_ylabel("Nombre")
        ax.set_ylim(0, ax.get_ylim()[1] + 200)  # increase the top by 200 counts
        ax.set_title("Niveau d'hydratation selon l'âge et le climat")

        # Legend (only hydration levels, not duplicated climates)
        ax.legend(title="Niveau d'hydratation")

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("""
        **Observations:**
        - On observe que les individus plus âgés ont tendance à avoir un taux de niveau d'hydratation "Poor" plus élevé, en particulier par temps froid.
        - Cela peut s'expliquer par le fait que la sensation de soif diminue avec l'âge, ce qui peut entraîner une consommation d'eau insuffisante chez les personnes âgées, surtout en climat froid où la sensation de soif est encore plus réduite.
        """)