"""Tests unitaires pour le calcul de la dérive (PSI)."""

import numpy as np
import pytest

from api.drift import population_stability_index

@pytest.mark.unitaire
def test_psi_on_identical_distributions():
    """
    Vérifie que le PSI est proche de zéro pour deux distributions identiques.
    Un PSI faible signifie qu'il n'y a pas de dérive de population.
    """
    # Distribution de référence
    reference_data = np.random.normal(loc=10, scale=2, size=1000)
    # Distribution actuelle (identique)
    current_data = reference_data.copy()

    psi = population_stability_index(reference_data, current_data)

    # Pour des distributions identiques, le PSI doit être très faible (proche de 0)
    assert psi < 0.01

@pytest.mark.unitaire
def test_psi_on_shifted_distributions():
    """
    Vérifie que le PSI est significatif (> 0.1) pour deux distributions différentes.
    Un PSI élevé signale une dérive de population.
    """
    # Distribution de référence
    reference_data = np.random.normal(loc=10, scale=2, size=1000)
    # Distribution actuelle avec une moyenne décalée
    current_data = np.random.normal(loc=12, scale=2, size=1000)

    psi = population_stability_index(reference_data, current_data)

    # Pour des distributions différentes, on s'attend à un PSI significatif
    assert psi > 0.1
