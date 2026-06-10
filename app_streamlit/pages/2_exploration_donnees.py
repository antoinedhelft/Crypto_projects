import streamlit as st

st.set_page_config(page_title="2 - Donnees", layout="wide")

st.title("2️⃣ Donnees et preparation")

st.markdown("Vue rapide du perimetre data utilise pour la demo.")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Source", "Binance API")
c2.metric("Granularite", "1h")
c3.metric("Donnees", "OHLCV")
c4.metric("Univers", "Top paires actives")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Perimetre", "Variables", "Qualite data"])

with tab1:
    st.subheader("Perimetre retenu")
    st.markdown(
        """
- Marche spot (ex: BTCUSDT, ETHUSDT).
- Historique exploite pour features et apprentissage.
- Mise a jour automatique des bougies.
        """
    )

with tab2:
    st.subheader("Variables principales")
    st.markdown(
        """
- Prix: open, high, low, close.
- Volumes: volume_base, volume_quote.
- Features derivees: lags, RSI, MACD, ATR, variables temporelles.
- Cibles: variation t+1h et direction (Baisse/Stable/Hausse).
        """
    )

with tab3:
    st.subheader("Points de controle qualite")
    st.markdown(
        """
- Fraicheur des dernieres bougies.
- Types et formats de colonnes.
- Detection des trous et doublons critiques.
- Verification de coherence avant prediction.
        """
    )

st.caption("Page 2 - Focus donnees pour demo")
