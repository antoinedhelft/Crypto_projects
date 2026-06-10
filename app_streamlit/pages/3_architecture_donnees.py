import streamlit as st
from pathlib import Path

st.set_page_config(page_title="3 - Architecture", layout="wide")

st.title("3️⃣ Architecture data")

st.markdown("Architecture simplifiee du pipeline, orientee exploitation.")

c1, c2, c3 = st.columns(3)
c1.metric("Stockage", "PostgreSQL")
c2.metric("Orchestration", "Airflow")
c3.metric("Exposition", "FastAPI")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Choix techniques")
    st.markdown(
        """
- Base relationnelle pour garantir coherence et requetabilite.
- Pipeline conteneurise pour reproductibilite.
- Separation ingestion / entrainement / serving.
        """
    )

with right:
    st.subheader("Flux de donnees")
    st.markdown(
        """
1. Ingestion des bougies.
2. Stockage et historisation SQL.
3. Feature engineering.
4. Entrainement et selection de modele.
5. Prediction via API et visualisation Streamlit.
        """
    )

root_dir = Path(__file__).parents[1]
img_path = root_dir / "images"
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    st.image(str(img_path / "uml.png"), width=900)

st.caption("Page 3 - Architecture courte pour demo")
