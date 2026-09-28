# AI Finance Copilot

Application métier de gestion financière conçue pour démontrer le passage d'un besoin métier à une solution IA exploitable.

## Démo

L'application est conçue pour être déployée sur Streamlit Community Cloud à partir de ce dépôt.

## Fonctionnalités

- Import CSV / Excel
- KPIs financiers
- Analyse des dépenses par catégorie et dans le temps
- Détection d'anomalies avec Isolation Forest
- Synthèse automatique
- Questions en langage naturel
- Mode démo fonctionnel sans clé API
- Intégration OpenAI optionnelle côté serveur
- Export des données filtrées

## Cas d'usage

Un responsable administratif ou financier importe un fichier de transactions et obtient immédiatement :
1. une vue consolidée de ses dépenses ;
2. les catégories qui concentrent les coûts ;
3. les transactions atypiques ;
4. une synthèse métier ;
5. un assistant permettant d'interroger les données simplement.

## Stack

Python, Streamlit, Pandas, Plotly, scikit-learn, OpenAI API.

## Lancer en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Activer le LLM

Créer `.streamlit/secrets.toml` :

```toml
OPENAI_API_KEY="votre_cle"
```

Ne jamais publier la clé dans GitHub.

## Format minimal des données

| date | categorie | montant |
|---|---|---:|
| 2026-01-03 | Logiciels | 129.90 |
| 2026-01-04 | Marketing | 520.00 |

Les colonnes `mode_paiement` et `statut` sont optionnelles.

## Architecture

```text
ai-finance-copilot/
├── app.py
├── data/
│   └── sample_transactions.csv
├── requirements.txt
├── README.md
└── .gitignore
```

## Valeur métier

Ce POC montre comment une petite équipe peut transformer un export de transactions en outil de pilotage interactif et y ajouter une couche d'IA pour accélérer l'analyse et la prise de décision.

## Auteur

Khadidiatou Kenewy Diallo
