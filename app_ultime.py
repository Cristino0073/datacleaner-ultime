import streamlit as st
import pandas as pd
import numpy as np
import re
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

    def log(self, msg): self.logs.append(msg)

    def semantic_standardization(self):
        """BRAIN: Renomme intelligemment les colonnes"""
        mapping = {}
        for col in self.df.columns:
            c = col.lower()
            if any(x in c for x in ['prx','prix','price','montant','amount']): mapping[col] = 'Prix'
            elif any(x in c for x in ['qte','qty','quantite']): mapping[col] = 'Quantite'
            elif any(x in c for x in ['prod','article','designation']): mapping[col] = 'Produit'
            elif any(x in c for x in ['dat','jour']): mapping[col] = 'Date'
            elif any(x in c for x in ['cli','nom']): mapping[col] = 'Client'
        self.df.rename(columns=mapping, inplace=True)
        self.log(f"Standardisation sémantique: {mapping}")

    def calculate_quality_score(self, df):
        score = 100
        score -= df.isnull().sum().sum() / (df.shape[0]*df.shape[1]) * 40
        score -= df.duplicated().sum() / df.shape[0] * 30
        return round(max(0, score), 1)

    def run_full_pipeline(self):
        self.quality_before = self.calculate_quality_score(self.df)
        self.semantic_standardization()

        # 1. Dates
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'], errors='coerce')
            self.df['Mois'] = self.df['Date'].dt.month
            self.df['Annee'] = self.df['Date'].dt.year
            self.log("Feature Factory: Mois et Année créés à partir de Date")

        # 2. Nettoyage Texte
        for col in self.df.select_dtypes(include='object').columns:
            self.df[col] = self.df[col].astype(str).str.strip().str.title().replace({'Nan':'Inconnu', 'None':'Inconnu'})

        # 3. Outliers & Doublons
        initial_rows = self.df.shape[0]
        self.df.drop_duplicates(inplace=True)
        for col in self.df.select_dtypes(include=[np.number]).columns:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            self.df = self.df[~((self.df[col] < Q1 - 3*IQR) | (self.df[col] > Q3 + 3*IQR))]
        self.log(f"{initial_rows - self.df.shape[0]} lignes aberrantes/doublons supprimés")

        # 4. Imputation Intelligente KNN
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1 and self.df[numeric_cols].isnull().sum().sum() > 0:
            imputer = KNNImputer(n_neighbors=3)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])
            self.log("Imputation KNN appliquée sur les colonnes numériques")

        # 5. Feature Business
        if 'Prix' in self.df.columns and 'Quantite' in self.df.columns:
            self.df['CA_Total'] = self.df['Prix'] * self.df['Quantite']
            self.log("Feature Business: CA_Total = Prix * Quantite créé")

        self.quality_after = self.calculate_quality_score(self.df)
        return self.df

# --- UI ---
file = st.file_uploader("📤 Dépose ton fichier brut (Excel/CSV)", type=['csv','xlsx'])

if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    agent = UltimateDataAgent(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes Brutes", df.shape[0])
    c2.metric("Colonnes", df.shape[1])
    c3.metric("Score Qualité Brut", f"{agent.calculate_quality_score(df)}/100")

    st.dataframe(df.head(3))

    if st.button("🔥 LANCER L'AGENT ULTIME", type="primary"):
        with st.spinner("L'agent analyse, nettoie, enrichit..."):
            df_clean = agent.run_full_pipeline()

        st.success(f"Terminé! Qualité passée de {agent.quality_before} -> {agent.quality_after}/100")

        tab1, tab2, tab3 = st.tabs(["✅ Données Propres", "📊 Dashboard Auto", "📝 Logs & Rapport"])

        with tab1:
            st.dataframe(df_clean)
            b1 = io.BytesIO(); df_clean.to_excel(b1, index=False)
            st.download_button("📥 Télécharger Excel Propre", b1.getvalue(), "ULTIME_donnees_propres.xlsx")

        with tab2:
            if 'CA_Total' in df_clean.columns:
                fig = px.bar(df_clean.groupby('Produit')['CA_Total'].sum().reset_index().head(10), x='Produit', y='CA_Total', title="Top 10 CA par Produit")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.text("\n".join(agent.logs))
            pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial",'',10)
            pdf.cell(0,10,f"Rapport Agent ULTIME - Qualite {agent.quality_before} -> {agent.quality_after}", ln=True)
            for l in agent.logs: pdf.multi_cell(0,6,l)
            b2 = io.BytesIO(pdf.output())
            st.download_button("📄 Télécharger Rapport PDF Certifié", b2.getvalue(), "rapport_ultime.pdf")
else:
    st.info("👆 Upload un fichier pour voir l'agent à l'œuvre")