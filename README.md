# Multi-Model Options Pricer & Greeks Analyzer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)

## [https://multimodeloptionpricer.streamlit.app/#3-surface-de-prix-black-scholes-prix-vs-spot-and-temps]

## Description
Une application quantitative interactive permettant de valoriser simplement des options financières (Européennes et Américaines) en utilisant des données de marché réelles récupérées via l'API Yahoo Finance. 

## Fonctionnalités & Mathématiques
* **Market Data :** Intégration de `yfinance` pour le calcul dynamique de la volatilité historique annualisée (log-rendements sur 252 jours).
* **Moteurs de Pricing :**
  * Formules fermées de **Black-Scholes-Merton**.
  * **Simulation de Monte Carlo** avec technique de réduction de variance (variables antithétiques) pour optimiser le temps de calcul.
  * **Arbres Binomiaux** (Modèle CRR) pour la valorisation des options américaines.
* **Analyse de Risque :** Calcul des sensibilités du portefeuille (Les Grecques : Delta, Gamma, Theta, nu, rho).
* **Data Visualization :** Tracé dynamique de la surface de prix (Plotly 3D) en fonction du sous-jacent et de la maturité.

## Exécution en local
```bash
git clone [https://github.com/locheyeloi/multi_model_option_pricer.git](https://github.com/locheyeloi/multi_model_option_pricer.git)
cd multi_model_option_pricer
pip install -r requirements.txt
streamlit run app.py
