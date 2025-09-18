import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
import cvxpy as cp
import streamlit as st
from io import BytesIO

# Configuration Streamlit
st.set_page_config(page_title="Optimiseur de Portefeuille", layout="centered")

#  Image d'en-tête
st.image("header.webp", use_column_width=True)
st.title("Simulateur de Portefeuille Intelligent")

#  Initialiser session_state
for key in ["price_data", "returns", "weights", "expected_return", "expected_risk"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Chargement du fichier
st.markdown("####  Charger un fichier CSV/XLS/XLSX contenant les données historiques (colonnes: Date + symboles)")
uploaded_file = st.file_uploader("Glisser ou sélectionner un fichier", type=["csv", "xls", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, parse_dates=["Date"], index_col="Date")
        else:
            df = pd.read_excel(uploaded_file, parse_dates=["Date"], index_col="Date")
        st.session_state.price_data = df
        st.success(" Fichier chargé avec succès.")
    except Exception as e:
        st.error(f" Erreur lors du chargement : {e}")

#  Choix du profil
profil = st.selectbox("Choisissez votre profil d'investisseur", ["Prudent", "Modéré", "Dynamique"])
risk_dict = {"Prudent": 10.0, "Modéré": 1.0, "Dynamique": 0.1}
risk_aversion = risk_dict[profil]

st.info({
    "Prudent": "💡 Vous privilégiez la sécurité à la performance.",
    "Modéré": "💡 Vous cherchez un bon équilibre entre risque et rendement.",
    "Dynamique": "💡 Vous acceptez plus de volatilité pour viser un rendement plus élevé."
}[profil])

#  Optimisation
if st.button("📊 Optimiser le portefeuille") and st.session_state.price_data is not None:
    price_data = st.session_state.price_data
    returns = price_data.pct_change().dropna() 
    mean_returns = returns.mean()  #  rendement moyen de chaque actif
    cov_matrix = returns.cov()

    n = len(price_data.columns) # n = nombre d'actifs (ex : 4 actions)
    w = cp.Variable(n)           # w = variable de décision (les poids de chaque actif)
    ret = mean_returns.values @ w  # le rendement total attendu du portefeuille 
    risk = cp.quad_form(w, cov_matrix.values) # risk = combien ton portefeuille peut varier (volatilité) /= vecteur des poids des actifs*matrice de covariance entre les actifs* transposée de w

    objective = cp.Maximize(ret - risk_aversion * risk)
    constraints = [cp.sum(w) == 1, w >= 0.05, w <= 0.8]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    weights = w.value
    expected_return = float(mean_returns.values @ weights)
    expected_risk = float(np.sqrt(weights.T @ cov_matrix.values @ weights))

    #  Stocker les résultats dans session_state
    st.session_state.returns = returns
    st.session_state.weights = weights
    st.session_state.expected_return = expected_return
    st.session_state.expected_risk = expected_risk

#  Affichage des résultats (même après modification du capital)
if st.session_state.weights is not None:

    price_data = st.session_state.price_data
    returns = st.session_state.returns
    weights = st.session_state.weights
    expected_return = st.session_state.expected_return
    expected_risk = st.session_state.expected_risk

    weights_df = pd.DataFrame({
        "Actif": price_data.columns,
        "Poids optimal (%)": np.round(weights * 100, 2)
    })

    st.subheader("📋 Répartition optimale du portefeuille")
    st.dataframe(weights_df)
    st.success(f"📈 Rendement attendu : {expected_return:.2%}")#le rendement moyen du portefeuille optimisé.
    st.warning(f"📉 Risque (volatilité) : {expected_risk:.2%}")#le risque total (volatilité) de ce portefeuille.

    fig, ax = plt.subplots()
    ax.pie(weights, labels=price_data.columns, autopct='%1.1f%%', startangle=140)
    ax.set_title("Répartition du portefeuille optimal")
    st.pyplot(fig)

    st.subheader("📊 Rendement cumulé du portefeuille")
    weighted_returns = (returns * weights).sum(axis=1)
    cumulative_returns = (1 + weighted_returns).cumprod() - 1
    st.line_chart(cumulative_returns)

    # Projection capital jusqu’en 2028
    st.subheader("📅 Projection du capital jusqu'en 2028")
    capital_initial = st.number_input("Capital initial (en DH)", min_value=1000.0, value=10000.0, step=100.0)

    nb_jours = (datetime(2028, 1, 1) - returns.index[-1]).days
    rendement_journalier = (1 + expected_return) ** (1 / 252) - 1
    vol_journalière = np.std(weighted_returns)

    np.random.seed(42)
    daily_returns = np.random.normal(loc=rendement_journalier, scale=vol_journalière, size=nb_jours)
    capital_projection = [capital_initial]
    for r in daily_returns:
        capital_projection.append(capital_projection[-1] * (1 + r))

    dates_proj = pd.date_range(start=returns.index[-1], periods=nb_jours + 1)
    projection_df = pd.Series(capital_projection, index=dates_proj)
    st.line_chart(projection_df)
    
    
        #  Simulation Monte Carlo du capital
    st.subheader("🎲 Simulation Monte Carlo du capital (1000 scénarios)")

    nb_simulations = 1000  #  Nombre de trajectoires à simuler

    rendement_journalier = (1 + expected_return) ** (1/252) - 1
    volatilite_journaliere = np.std(weighted_returns)

    np.random.seed(42)  # 🌱 Pour avoir les mêmes résultats à chaque exécution
    simulations = np.zeros((nb_jours + 1, nb_simulations))  # Tableau vide

    for i in range(nb_simulations): # Boucle pour faire plusieurs simulations #Chaque tour i génère un scénario complet d'évolution du capital sur plusieurs jours.
        daily_returns = np.random.normal(loc=rendement_journalier, scale=volatilite_journaliere, size=nb_jours)#un tableau de rendements quotidiens simulés pour 1 simulation.
        #np.random.normal()	Générer les rendements journaliers au hasard (mais réalistes).
        capital_path = [capital_initial]#créer une liste qui va contenir l'évolution du capital : On commence la trajectoire avec ce capital initial. 
        
        #Calculer jour après jour l'évolution du capital :
        for r in daily_returns: # Pour chaque rendement r simulé du tableau daily_returns
            capital_path.append(capital_path[-1] * (1 + r)) # capital_path[-1] ➔ prend le dernier capital connu. Tu construis toute la trajectoire du capital, jour après jour
            
        simulations[:, i] = capital_path #Stocker la trajectoire complète dans le tableau simulations chaque colonne i reçoit la trajectoire complète du capital pour la simulation numéro i.
        # ➔ À la fin, tu auras nb_simulations colonnes (ex : 1000 trajectoires).
        #simulations, c’est ton grand tableau où :
        #Chaque colonne = 1 simulation complète du capital sur tous les jours.
        #Chaque ligne = le capital à une date donnée dans toutes les simulations.
        
    dates_proj = pd.date_range(start=returns.index[-1], periods=nb_jours + 1) #Créer les dates associées aux simulations
    #start=returns.index[-1] ➔ on commence à la dernière date connue de ton historique réel.
    #periods=nb_jours + 1 ➔ un jour de départ + tous les jours simulés.

    #  Afficher toutes les trajectoires
    fig, ax = plt.subplots(figsize=(12, 6))
    #Créer une figure vide :
#fig ➔ la figure principale (l’espace global) 
#ax ➔ les axes pour tracer (x = temps, y = capital). 
#figsize=(12,6) ➔ largeur 12 pouces, hauteur 6 pouces ➔ un graphique large et lisible.


    ax.plot(dates_proj, simulations, color="skyblue", alpha=0.05)  # Tracer toutes les trajectoires simulées :

    #simulations.mean(axis=1) ➔ calcule la moyenne du capital jour par jour.#Pour chaque jour, faire la moyenne de tous les capitaux sur toutes les simulations.
    #linewidth=2 ➔ plus épais pour que la moyenne soit bien visible
    ax.plot(dates_proj, simulations.mean(axis=1), color="blue", label="Capital moyen", linewidth=2)  # 🔵 moyenne
    ax.set_title("Projection Monte Carlo du capital jusqu'en 2028")
    
    
    #Nommer les axes :
    ax.set_xlabel("Date")
    ax.set_ylabel("Capital (DH)")
    ax.legend()
    ax.grid(True)
    
    # dit à Streamlit d’afficher la figure matplotlib dans l’application
    st.pyplot(fig)

    #  Résumé rapide
    st.info(f"📈 Capital moyen en 2028 : {simulations[-1].mean():,.2f} DH")
    st.info(f"📉 Capital minimum observé en 2028 : {simulations[-1].min():,.2f} DH")
    st.info(f"📈 Capital maximum observé en 2028 : {simulations[-1].max():,.2f} DH") 
    
    
    #Quand tu as simulé 1000 trajectoires du capital ➔
     #à la fin, en 2028, tu as 1000 résultats différents (1000 capitaux possibles).
    # Calcul de l'intervalle de confiance 80% (entre 10% et 90%)
    capital_inf = np.percentile(simulations[-1], 10)  # 10ème percentile #10e percentile = la valeur sous laquelle tombent les 10% plus faibles ➔ limite basse (capital_inf)

    #90e percentile = la valeur sous laquelle tombent 90% des cas ➔ limite haute (capital_sup)
    capital_sup = np.percentile(simulations[-1], 90)  # 90ème percentile
  
#  Affichage dans Streamlit
         
    st.success(f"🔵 Avec 80% de probabilité, votre capital en 2028 sera compris entre {capital_inf:,.2f} DH et {capital_sup:,.2f} DH.")



    #  Export Excel
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        weights_df.to_excel(writer, index=False, sheet_name="Portefeuille")
    st.download_button(
        label="📥 Télécharger la répartition en Excel",
        data=output.getvalue(),
        file_name="portefeuille_optimise.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

#  Message de bas de page
st.markdown("""
---
**Cette plateforme a été conçue uniquement à des fins pédagogiques.  
Elle ne constitue pas un outil d’investissement professionnel.**
""")
