
import os
import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="AI Finance Copilot", page_icon="💼", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
div[data-testid="stMetric"] {background:#ffffff; border:1px solid #e9ecef; padding:16px; border-radius:14px;}
.small-note {color:#6b7280; font-size:.9rem;}
</style>
""", unsafe_allow_html=True)

st.title("💼 AI Finance Copilot")
st.caption("Assistant intelligent de gestion financière — analyse, anomalies, synthèses et questions métier.")

@st.cache_data
def load_sample():
    return pd.read_csv("data/sample_transactions.csv", parse_dates=["date"])

def normalize_columns(df):
    mapping = {}
    for c in df.columns:
        lc = c.lower().strip()
        if lc in ["date","transaction_date","jour"]: mapping[c] = "date"
        elif lc in ["montant","amount","valeur","total"]: mapping[c] = "montant"
        elif lc in ["categorie","catégorie","category","type"]: mapping[c] = "categorie"
        elif lc in ["mode_paiement","payment_method","paiement"]: mapping[c] = "mode_paiement"
        elif lc in ["statut","status"]: mapping[c] = "statut"
    df = df.rename(columns=mapping)
    return df

def validate_df(df):
    required = {"date","categorie","montant"}
    return required.issubset(set(df.columns))

def kpi_text(df):
    total = df["montant"].sum()
    avg = df["montant"].mean()
    max_row = df.loc[df["montant"].idxmax()]
    top_cat = df.groupby("categorie")["montant"].sum().sort_values(ascending=False).index[0]
    return total, avg, max_row, top_cat

def detect_anomalies(df):
    tmp = df.copy()
    x = tmp[["montant"]].fillna(tmp["montant"].median())
    if len(tmp) < 20:
        tmp["anomalie"] = False
        tmp["score_anomalie"] = 0.0
        return tmp
    model = IsolationForest(contamination=min(0.05, max(0.01, 10/len(tmp))), random_state=42)
    pred = model.fit_predict(x)
    scores = -model.score_samples(x)
    tmp["anomalie"] = pred == -1
    tmp["score_anomalie"] = scores
    return tmp

def local_summary(df):
    total, avg, max_row, top_cat = kpi_text(df)
    monthly = df.assign(mois=df["date"].dt.to_period("M").astype(str)).groupby("mois")["montant"].sum()
    trend = "stable"
    if len(monthly) >= 2:
        change = (monthly.iloc[-1] - monthly.iloc[-2]) / max(monthly.iloc[-2], 1) * 100
        trend = f"{change:+.1f}% par rapport au mois précédent"
    return (
        f"Les dépenses analysées totalisent {total:,.0f} € pour un montant moyen de {avg:,.0f} € par transaction. "
        f"La catégorie la plus importante est « {top_cat} ». "
        f"L'évolution du dernier mois est de {trend}. "
        f"La transaction la plus élevée est de {max_row['montant']:,.0f} € dans la catégorie « {max_row['categorie']} »."
    ).replace(",", " ")

def answer_local(df, question):
    q = question.lower()
    total, avg, max_row, top_cat = kpi_text(df)
    if any(k in q for k in ["total", "dépense totale", "depense totale"]):
        return f"Le montant total des dépenses est de {total:,.2f} €.".replace(",", " ")
    if any(k in q for k in ["catégorie", "categorie", "plus dépens", "plus depens"]):
        by_cat = df.groupby("categorie")["montant"].sum().sort_values(ascending=False)
        return f"La catégorie la plus coûteuse est « {by_cat.index[0]} » avec {by_cat.iloc[0]:,.2f} €.".replace(",", " ")
    if any(k in q for k in ["moyen", "moyenne"]):
        return f"Le montant moyen par transaction est de {avg:,.2f} €.".replace(",", " ")
    if any(k in q for k in ["maximum", "plus grosse", "plus élevé", "plus eleve"]):
        return f"La transaction la plus élevée est de {max_row['montant']:,.2f} € le {max_row['date'].date()} dans « {max_row['categorie']} ».".replace(",", " ")
    if any(k in q for k in ["anomal", "suspect", "inhabit"]):
        adf = detect_anomalies(df)
        n = int(adf["anomalie"].sum())
        return f"J'ai détecté {n} transaction(s) atypique(s) avec Isolation Forest. Consulte l'onglet « Anomalies » pour le détail."
    return local_summary(df)

def openai_answer(df, question):
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        context = df.head(250).to_csv(index=False)
        prompt = f"""Tu es un copilote financier pour PME.
Réponds uniquement à partir des données CSV ci-dessous.
Sois précis, concis et orienté métier. Si la donnée manque, dis-le.
Question: {question}

Données:
{context}
"""
        resp = client.responses.create(model="gpt-4.1-mini", input=prompt)
        return resp.output_text
    except Exception:
        return None

with st.sidebar:
    st.header("Données")
    source = st.radio("Source", ["Jeu de démonstration", "Importer un fichier"])
    uploaded = None
    if source == "Importer un fichier":
        uploaded = st.file_uploader("CSV ou Excel", type=["csv","xlsx","xls"])
    st.markdown("---")
    st.markdown("**Colonnes attendues**")
    st.code("date | categorie | montant")
    st.caption("Colonnes optionnelles : mode_paiement, statut")

if source == "Jeu de démonstration":
    df = load_sample()
elif uploaded is not None:
    if uploaded.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)
    df = normalize_columns(df)
else:
    st.info("Importez un fichier CSV ou Excel pour commencer.")
    st.stop()

df = normalize_columns(df)
if not validate_df(df):
    st.error("Le fichier doit contenir au minimum les colonnes : date, categorie, montant.")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["montant"] = pd.to_numeric(df["montant"], errors="coerce")
df = df.dropna(subset=["date","montant","categorie"]).copy()

min_d, max_d = df["date"].min().date(), df["date"].max().date()
with st.expander("🔎 Filtres", expanded=False):
    c1, c2 = st.columns(2)
    with c1:
        selected_dates = st.date_input("Période", value=(min_d, max_d), min_value=min_d, max_value=max_d)
    with c2:
        cats = st.multiselect("Catégories", sorted(df["categorie"].astype(str).unique()), default=sorted(df["categorie"].astype(str).unique()))
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start, end = pd.Timestamp(selected_dates[0]), pd.Timestamp(selected_dates[1])
        df = df[(df["date"] >= start) & (df["date"] <= end)]
    if cats:
        df = df[df["categorie"].isin(cats)]

if df.empty:
    st.warning("Aucune donnée avec ces filtres.")
    st.stop()

total, avg, max_row, top_cat = kpi_text(df)
a, b, c, d = st.columns(4)
a.metric("Dépenses totales", f"{total:,.0f} €".replace(",", " "))
b.metric("Transaction moyenne", f"{avg:,.0f} €".replace(",", " "))
c.metric("Nb. transactions", f"{len(df):,}".replace(",", " "))
d.metric("Catégorie principale", top_cat)

tabs = st.tabs(["📊 Dashboard", "🚨 Anomalies", "🤖 Copilote IA", "📥 Données"])

with tabs[0]:
    monthly = df.assign(mois=df["date"].dt.to_period("M").astype(str)).groupby("mois", as_index=False)["montant"].sum()
    by_cat = df.groupby("categorie", as_index=False)["montant"].sum().sort_values("montant", ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.line(monthly, x="mois", y="montant", markers=True, title="Évolution mensuelle"), use_container_width=True)
    with c2:
        st.plotly_chart(px.bar(by_cat, x="categorie", y="montant", title="Dépenses par catégorie"), use_container_width=True)
    st.subheader("Synthèse automatique")
    st.success(local_summary(df))

with tabs[1]:
    adf = detect_anomalies(df)
    anomalies = adf[adf["anomalie"]].sort_values("score_anomalie", ascending=False)
    st.metric("Transactions atypiques détectées", len(anomalies))
    if len(anomalies):
        st.dataframe(anomalies[["date","categorie","montant","score_anomalie"]], use_container_width=True)
        st.caption("Détection non supervisée par Isolation Forest. Une anomalie n'implique pas nécessairement une fraude.")
    else:
        st.info("Aucune anomalie significative détectée.")

with tabs[2]:
    st.subheader("Posez une question sur les données")
    st.caption("Mode démo utilisable sans clé API. Si une clé OpenAI est configurée côté serveur, le LLM est utilisé automatiquement.")
    q = st.text_input("Exemples : Quelle catégorie coûte le plus ? Combien avons-nous dépensé ? Y a-t-il des anomalies ?")
    if q:
        ans = openai_answer(df, q)
        if ans:
            st.markdown("### Réponse LLM")
            st.write(ans)
        else:
            st.markdown("### Réponse du moteur local")
            st.write(answer_local(df, q))
    st.markdown("#### Questions rapides")
    cols = st.columns(4)
    examples = [
        "Quelle catégorie coûte le plus ?",
        "Quel est le total des dépenses ?",
        "Quelle est la transaction moyenne ?",
        "Y a-t-il des anomalies ?"
    ]
    for i, ex in enumerate(examples):
        if cols[i].button(ex, use_container_width=True):
            st.write(answer_local(df, ex))

with tabs[3]:
    st.dataframe(df.sort_values("date", ascending=False), use_container_width=True)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Télécharger les données filtrées", csv, "transactions_filtrees.csv", "text/csv")

st.markdown("---")
st.markdown(
    "<div class='small-note'>Projet portfolio — Python · Streamlit · Pandas · Plotly · scikit-learn · OpenAI API (optionnelle)</div>",
    unsafe_allow_html=True
)
