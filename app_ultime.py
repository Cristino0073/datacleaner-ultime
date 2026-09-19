import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from fpdf import FPDF
import io
import plotly.express as px

st.set_page_config(page_title="DataCleaner ULTIME 4.0 - Cristino", layout="wide", page_icon="🚀")
st.title("🚀 DataCleaner Agent ULTIME 4.0")
st.caption("L'agent qui transforme le chaos en données prêtes pour la décision | By Cristino VELOMODY")

# --- CERVEAU DE L'AGENT ---
class UltimateDataAgent:
    def __init__(self, df):
        self.df = df.copy()
        self.df_original = df.copy()
        self.logs = []
        self.quality_before = 0
        self.quality_after = 0

    def log(self, msg):
        self.logs.append(msg)

    def semantic_standardization(self):
        """BRAIN: Renomme intelligemment les colonnes"""
        mapping = {}
        seen_targets = {}

        for col in self.df.columns:
            c = str(col).lower()

            if any(x in c for x in ['prx', 'prix', 'price', 'montant', 'amount']):
                target = 'Prix'
            elif any(x in c for x in ['qte', 'qty', 'quantite']):
                target = 'Quantite'
            elif any(x in c for x in ['prod', 'article', 'designation', 'libelle', 'nom_produit']):
                target = 'Produit'
            elif any(x in c for x in ['dat', 'jour', 'date']):
                target = 'Date'
            elif any(x in c for x in ['cli', 'client', 'nom_client']):
                target = 'Client'
            else:
                target = None

            if target is not None:
                seen_targets[target] = seen_targets.get(target, 0) + 1
                if seen_targets[target] > 1:
                    target = f"{target}_{seen_targets[target]}"
                mapping[col] = target

        self.df.rename(columns=mapping, inplace=True)
        self.log(f"Standardisation sémantique: {mapping}")

    def calculate_quality_score(self, df):
        if df.empty:
            return 0.0

        score = 100
        if df.shape[0] * df.shape[1] > 0:
            score -= df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 40
        if df.shape[0] > 0:
            score -= df.duplicated().sum() / df.shape[0] * 30
        return round(max(0, score), 1)

    def run_full_pipeline(self):
        self.quality_before = self.calculate_quality_score(self.df)

        if self.df.empty:
            self.log("Le DataFrame est vide, rien à traiter.")
            self.quality_after = 0
            return self.df

        self.semantic_standardization()

        # Nettoyage des types de base
        for col in self.df.columns:
            if col in ['Date', 'Mois', 'Annee']:
                continue
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].astype(str).str.strip()

        # 1. Dates
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            if self.df['Date'].notna().any():
                self.df['Mois'] = self.df['Date'].dt.month
                self.df['Annee'] = self.df['Date'].dt.year
                self.log("Feature Factory: Mois et Année créés à partir de Date")

        # 2. Nettoyage Texte
        text_cols = self.df.select_dtypes(include=['object', 'string']).columns.tolist()
        for col in text_cols:
            if col in ['Date', 'Mois', 'Annee']:
                continue
            self.df[col] = self.df[col].replace({np.nan: 'Inconnu', None: 'Inconnu', 'nan': 'Inconnu', 'None': 'Inconnu'})
            self.df[col] = self.df[col].astype(str).str.strip()
            self.df[col] = self.df[col].apply(
                lambda x: x.title() if x and x.lower() not in ['inconnu', 'nan', 'none', ''] else x
            )

        # 3. Conversion des colonnes métier
        for col in ['Prix', 'Quantite']:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # 4. Outliers & Doublons
        initial_rows = self.df.shape[0]
        self.df = self.df.drop_duplicates()

        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if self.df[col].dropna().empty:
                continue
            if self.df[col].dropna().nunique() < 2:
                continue

            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            if pd.isna(Q1) or pd.isna(Q3) or pd.isna(IQR) or IQR == 0:
                continue

            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR

            self.df = self.df[~((self.df[col] < lower_bound) | (self.df[col] > upper_bound))]

        self.log(f"{initial_rows - self.df.shape[0]} lignes aberrantes/doublons supprimés")

        # 5. Imputation KNN
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2 and self.df[numeric_cols].isnull().sum().sum() > 0:
            valid_rows = self.df[numeric_cols].dropna().shape[0]
            if valid_rows > 1:
                n_neighbors = min(3, max(1, valid_rows))
                imputer = KNNImputer(n_neighbors=n_neighbors)
                self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
                self.log("Imputation KNN appliquée sur les colonnes numériques")

        # 6. Feature Business
        if 'Prix' in self.df.columns and 'Quantite' in self.df.columns:
            self.df['CA_Total'] = self.df['Prix'].fillna(0) * self.df['Quantite'].fillna(0)
            self.log("Feature Business: CA_Total = Prix * Quantite créé")

        self.quality_after = self.calculate_quality_score(self.df)
        return self.df

# --- UI ---
file = st.file_uploader("📤 Dépose ton fichier brut (Excel/CSV)", type=['csv', 'xlsx'])

if file:
    try:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, sep=None, engine='python')
        else:
            df = pd.read_excel(file)
    except Exception as e:
        st.error(f"Impossible de lire le fichier : {e}")
        st.stop()

    agent = UltimateDataAgent(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes Brutes", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Score Qualité Brut", f"{agent.calculate_quality_score(df)}/100")

    st.dataframe(df.head(5))

    if st.button("🔥 LANCER L'AGENT ULTIME", type="primary"):
        with st.spinner("L'agent analyse, nettoie, enrichit..."):
            df_clean = agent.run_full_pipeline()

        if df_clean.empty:
            st.warning("Le jeu de données est devenu vide après nettoyage. Vérifie les colonnes ou le format du fichier.")
        else:
            st.success(f"Terminé! Qualité passée de {agent.quality_before} -> {agent.quality_after}/100")

            tab1, tab2, tab3 = st.tabs(["✅ Données Propres", "📊 Dashboard Auto", "📝 Logs & Rapport"])

            with tab1:
                st.dataframe(df_clean)
                b1 = io.BytesIO()
                df_clean.to_excel(b1, index=False)
                st.download_button("📥 Télécharger Excel Propre", b1.getvalue(), "ULTIME_donnees_propres.xlsx")

            with tab2:
                if 'CA_Total' in df_clean.columns:
                    fig = px.bar(
                        df_clean.groupby('Produit', dropna=False)['CA_Total'].sum().reset_index().head(10),
                        x='Produit', y='CA_Total', title="Top 10 CA par Produit"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucune colonne 'Produit' ou 'CA_Total' trouvée pour générer le dashboard.")

            with tab3:
                st.text("\n".join(agent.logs) if agent.logs else "Aucun log à afficher.")
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", '', 10)
                pdf.cell(0, 10, f"Rapport Agent ULTIME - Qualite {agent.quality_before} -> {agent.quality_after}", ln=True)
                for l in agent.logs:
                    pdf.multi_cell(0, 6, l)
                b2 = io.BytesIO(pdf.output())
                st.download_button("📄 Télécharger Rapport PDF Certifié", b2.getvalue(), "rapport_ultime.pdf")
else:
    st.info("👆 Upload un fichier pour voir l'agent à l'œuvre")
