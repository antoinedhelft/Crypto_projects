import streamlit as st
from pathlib import Path

st.set_page_config(page_title="3 - Architecture & organisation des données", layout="wide")

st.title("3️⃣ Architecture et organisation des données")

tabs = st.tabs([
    "3.1 Choix base relationnelle (SQL vs NoSQL)",
    "3.2 Schéma UML & tables (markets, pairs, candlesticks, trades, crypto, exchange)",
    
])

with tabs[0]:
    st.header("3.1 Choix d’une base relationnelle (SQL vs NoSQL – justification)")

    root_dir = Path(__file__).parents[1]
    img_path = root_dir / "images"

    # Centrer l'image avec des colonnes Streamlit
    col_l,col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(str(img_path / "choixsql.png"), width=1000)

with tabs[1]:
    st.header("3.2 Schéma UML et explication des tables (markets, pairs, candlesticks, trades, crypto, exchange)")

    root_dir = Path(__file__).parents[1]
    img_path = root_dir / "images"

    # Centrer l'image avec des colonnes Streamlit
    col_l,col_c, col_r = st.columns([1, 2, 1])
    with col_c:
        st.image(str(img_path / "uml.png"), width=1000)


st.caption("Page 3 – Architecture & data")
