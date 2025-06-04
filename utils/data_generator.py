import numpy as np
import random
from typing import Tuple


def generate_linearly_separable_data(
    n_samples: int = 1000, n_features: int = 2, random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados sintéticos linearmente separáveis para classificação binária.

    Args:
        n_samples: Número de amostras a serem geradas
        n_features: Número de características (features)
        random_state: Seed para reprodutibilidade

    Returns:
        Tuple contendo:
        - X: Array de features com shape (n_samples, n_features)
        - y: Array de labels com shape (n_samples,) contendo 0s e 1s
    """
    if random_state is not None:
        np.random.seed(random_state)
        random.seed(random_state)

    # Gera duas classes gaussianas bem separadas
    n_samples_per_class = n_samples // 2

    # Classe 0: centrada em (-2, -2) para 2D ou valores negativos para n-D
    mean_0 = [-2.0] * n_features
    X_0 = np.random.multivariate_normal(
        mean_0, np.eye(n_features) * 0.5, n_samples_per_class
    )
    y_0 = np.zeros(n_samples_per_class)

    # Classe 1: centrada em (2, 2) para 2D ou valores positivos para n-D
    mean_1 = [2.0] * n_features
    X_1 = np.random.multivariate_normal(
        mean_1, np.eye(n_features) * 0.5, n_samples_per_class
    )
    y_1 = np.ones(n_samples_per_class)

    # Combina as duas classes
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    # Embaralha os dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


def generate_simple_2d_data(
    n_samples: int = 1000, random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera dados 2D simples e linearmente separáveis para demonstração.

    Args:
        n_samples: Número de amostras por classe
        random_state: Seed para reprodutibilidade

    Returns:
        Tuple contendo X (features) e y (labels)
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Classe 0: pontos no quadrante inferior esquerdo
    X_0 = np.random.uniform(-3, -0.5, (n_samples, 2))
    y_0 = np.zeros(n_samples)

    # Classe 1: pontos no quadrante superior direito
    X_1 = np.random.uniform(0.5, 3, (n_samples, 2))
    y_1 = np.ones(n_samples)

    # Combina e embaralha
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])

    indices = np.random.permutation(len(X))
    return X[indices], y[indices]
