import streamlit as st

st.set_page_config(page_title="2 - Exploration & manipulation des données", layout="wide")

st.title("2️⃣ Exploration et manipulation des données")

tabs = st.tabs([
    "2.1 Sources de données (Binance REST, WebSocket, Data Portal)",
    "2.2 Périmètre retenu & jeux de données",
    "2.3 Volumétrie & caractéristiques",
    "2.4 Variables pertinentes & cibles",
    "2.5 Prétraitements & feature engineering",
])

with tabs[0]:
    st.header("2.1 Sources de données identifiées (Binance REST API, WebSocket, Data Portal)")

    st.markdown(
    """
    
    - 📡 **Binance REST API** → données historiques OHLCV  
    - 🔄 **Binance WebSocket API** → données temps réel  
    - 📁 **Binance Data Portal** → historiques CSV publics   
    """)

with tabs[1]:
    st.header("2.2 Périmètre retenu et jeux de données utilisés")

    st.markdown(
    """
    
    - 🪙 Marché **spot Binance** (BTC/USDT, ETH/USDT, etc.)  
    - ⏱️ Intervalle temporel : **1h**  
    - 📊 Données : OHLCV (Open, High, Low, Close, Volume)  
    - 🤖 Sélection dynamique des **3 cryptos avec le plus de volume**  
  
    """)

with tabs[2]:
    st.header("2.3 Volumétrie et caractéristiques des données")

    st.markdown(
    """
    
    ✅ Points clés à afficher :
    - ⏱️ 8 760 lignes/an (1 ligne par heure)  
    - 📈 35 040 lignes pour 4 ans par paire  
    - 💾 ~20 à 30 Mo pour 3–4 paires sur 4 ans  
    - 🧩 Données tabulaires structurées : timestamps, prix, volumes, trades   
    """)

with tabs[3]:
    st.header("2.4 Variables pertinentes et variable(s) cible(s)")

    st.markdown(
    """
    
    Les principales variables exploitées sont :

    🕒 Temporelle
    - open_time, close_time
    - Début et fin de la période

    💰 Marché
    - open, high, low, close
    - Prix d’ouverture, haut, bas et fermeture

    📊 Activité
    - volume, number_of_trades
    - Volume échangé et nombre de transactions

💵 Takers
    - taker_buy_base_volume, taker_buy_quote_volume
    - Volumes achetés par les acteurs du marché
   
    """)

with tabs[4]:
    st.header("2.5 Prétraitements et feature engineering (nettoyage, normalisation, enrichissement)")

    st.markdown(
    """
    
    ### ✅ Points clés à afficher :
    - ✅ Données Binance déjà propres et exploitables  
    - 🕒 Conversion des timestamps → `datetime`  
    - 🔍 Vérification doublons / types / cohérence   
    """)

st.caption("Page 2 – Exploration des données")
