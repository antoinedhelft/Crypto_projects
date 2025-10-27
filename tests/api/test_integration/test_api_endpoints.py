"""Tests d'intégration pour les endpoints de l'API.

Ces tests nécessitent la présence de modèles et de listes de features dans MODELS_DIR.
Ils sont marqués @pytest.mark.integration et peuvent être ignorés en CI si les artefacts
ne sont pas disponibles.
"""

import pytest
from pathlib import Path
from api.settings import MODELS_DIR

# Vérifier la présence des artefacts nécessaires
has_models = any(MODELS_DIR.glob("*.joblib"))
has_reg_feats = any(MODELS_DIR.glob("regressor_features*.json"))
has_clf_feats = any(MODELS_DIR.glob("classifier_features*.json"))

REQUIRES_ARTIFACTS = pytest.mark.skipif(
    not (has_models and has_reg_feats and has_clf_feats),
    reason="Nécessite des artefacts réels (modèles et listes de features) dans MODELS_DIR"
)


@pytest.mark.integration
def test_health_endpoint(client):
    """Vérifie que l'endpoint /health est accessible et renvoie le bon statut."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


@pytest.mark.integration
def test_models_list_endpoint(client):
    """Vérifie que l'endpoint /models retourne une liste de modèles."""
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


@pytest.mark.integration
@REQUIRES_ARTIFACTS
def test_predict_price_only_endpoint(client):
    """Teste l'endpoint /predict_price_only avec un payload valide."""
    # Charger dynamiquement les features requises depuis les fichiers JSON
    import json
    reg_feats_files = sorted(MODELS_DIR.glob("regressor_features*.json"))
    if not reg_feats_files:
        pytest.skip("Aucun fichier regressor_features*.json trouvé")
    
    with open(reg_feats_files[-1], 'r') as f:
        reg_features = json.load(f)
    
    # Créer un payload avec des valeurs factices mais cohérentes
    payload = {feat: 0.0 for feat in reg_features}
    # Ajuster quelques valeurs pour être plus réalistes
    if "close_price" in payload:
        payload["close_price"] = 60000.0
    if "volume_base" in payload:
        payload["volume_base"] = 100.0
    
    response = client.post("/predict_price_only", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))


@pytest.mark.integration
@REQUIRES_ARTIFACTS
def test_predict_direction_only_endpoint(client):
    """Teste l'endpoint /predict_direction_only avec un payload valide."""
    import json
    clf_feats_files = sorted(MODELS_DIR.glob("classifier_features*.json"))
    if not clf_feats_files:
        pytest.skip("Aucun fichier classifier_features*.json trouvé")
    
    with open(clf_feats_files[-1], 'r') as f:
        clf_features = json.load(f)
    
    # Créer un payload avec des valeurs factices
    payload = {feat: 0.0 for feat in clf_features}
    if "rsi" in payload:
        payload["rsi"] = 50.0
    
    response = client.post("/predict_direction_only", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_direction_label" in data
    assert data["predicted_direction_label"] in ["Baisse", "Stable", "Hausse"]


@pytest.mark.integration
@REQUIRES_ARTIFACTS
def test_predict_price_only_missing_features(client):
    """Vérifie que l'endpoint /predict_price_only retourne une erreur si des features manquent."""
    # Envoyer un payload incomplet
    payload = {"close_price": 60000.0}
    
    response = client.post("/predict_price_only", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "manquantes" in data["detail"].lower()
