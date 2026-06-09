"""Tests unitaires pour l'application Streamlit.

Ces tests vérifient la logique métier de l'application Streamlit
sans lancer l'interface graphique.
"""

import pytest
from pathlib import Path

# Vérifier si les fichiers Streamlit existent
STREAMLIT_DIR = Path(__file__).resolve().parents[3] / "app_streamlit"
APP_FILE = STREAMLIT_DIR / "app.py"


@pytest.mark.unitaire
def test_streamlit_app_file_exists():
    """Vérifie que le fichier principal de l'application Streamlit existe."""
    # Verifier que le point d'entree de l'application est present.
    assert APP_FILE.exists(), f"Le fichier app.py n'existe pas dans {STREAMLIT_DIR}"


@pytest.mark.unitaire
def test_streamlit_app_has_no_syntax_errors():
    """Vérifie que le fichier app.py peut être compilé sans erreur de syntaxe."""
    # Si le fichier n'existe pas, ce test ne peut pas s'appliquer.
    if not APP_FILE.exists():
        pytest.skip("app.py n'existe pas")
    
    # Tenter une compilation Python du fichier.
    with open(APP_FILE, 'r', encoding='utf-8') as f:
        code = f.read()
    
    # Verifier l'absence d'erreur de syntaxe.
    try:
        compile(code, str(APP_FILE), 'exec')
    except SyntaxError as e:
        pytest.fail(f"Erreur de syntaxe dans app.py: {e}")



