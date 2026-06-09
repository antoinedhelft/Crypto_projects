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
    # Preparer deux distributions identiques.
    reference_data = np.random.normal(loc=10, scale=2, size=1000)
    current_data = reference_data.copy()

    # Calculer le PSI.
    psi = population_stability_index(reference_data, current_data)

    # Verifier que la derive est quasi nulle.
    assert psi < 0.01

@pytest.mark.unitaire
def test_psi_on_shifted_distributions():
    """
    Vérifie que le PSI est significatif (> 0.1) pour deux distributions différentes.
    Un PSI élevé signale une dérive de population.
    """
    # Preparer une distribution de reference et une distribution decalee.
    reference_data = np.random.normal(loc=10, scale=2, size=1000)
    current_data = np.random.normal(loc=12, scale=2, size=1000)

    # Calculer le PSI.
    psi = population_stability_index(reference_data, current_data)

    # Verifier que la derive est notable.
    assert psi > 0.1
