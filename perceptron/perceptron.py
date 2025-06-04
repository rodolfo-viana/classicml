import numpy as np
import random
from typing import Tuple, Optional


class Perceptron:
    """
    Implementação do algoritmo Perceptron de Rosenblatt (1957) para classificação binária.

    Trata-se de um algoritmo de aprendizado supervisionado que encontra um hiperplano
    linear para separar duas classes. Funciona apenas com dados linearmente separáveis.

    Equações Matemáticas:
    --------------------

    3. Regra de atualização de pesos (Delta Rule):


    4. Fronteira de decisão (para 2D):
       w₀ + w₁x₁ + w₂x₂ = 0
       Resolvendo para x₂: x₂ = -(w₀ + w₁x₁) / w₂

    Attributes:
        learning_rate: Taxa de aprendizado (default: 0.01)
        max_epochs: Número máximo de épocas de treinamento (default: 1000)
        random_state: Seed para reprodutibilidade (default: None)
        weights_: Pesos aprendidos (incluindo bias)
        n_epochs_: Número de épocas necessárias para convergência
        errors_: Histórico de erros por época
        is_fitted_: Indica se o modelo foi treinado
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        max_epochs: int = 1000,
        random_state: Optional[int] = None,
    ):
        """
        Inicializa o Perceptron.

        Args:
            learning_rate: Taxa de aprendizado para atualização dos pesos
            max_epochs: Número máximo de épocas de treinamento
            random_state: Seed para reprodutibilidade
        """
        # Hiperparâmetros
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state

        # Parâmetros
        self.weights_ = None
        self.n_epochs_ = 0
        self.errors_ = []
        self.is_fitted_ = False

    def _initialize_weights(self, n_features: int) -> None:
        """
        Inicializa os pesos com valores pequenos aleatórios.

        Args:
            n_features: Número de features de entrada
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        # Inicializa pesos com valores pequenos aleatórios
        # +1 para incluir o bias (peso do termo constante)
        self.weights_ = np.random.uniform(-0.5, 0.5, n_features + 1)

    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """
        Adiciona o termo de bias (coluna de 1s) à matriz de features.

        Args:
            X: Matriz de features

        Returns:
            Matriz de features com bias adicionado
        """
        return np.column_stack([np.ones(X.shape[0]), X])

    def _activation_function(self, z: np.ndarray) -> np.ndarray:
        """
        Função de ativação degrau.

        Equação:
        --------
        ŷ = φ(z) = { 1 se z ≥ 0
                   { 0 se z < 0

        onde z é a soma ponderada.

        Args:
            z: Soma ponderada

        Returns:
            Saída binária (0 ou 1)
        """
        return np.where(z >= 0, 1, 0)

    def _net_input(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a soma ponderada (net input) para as amostras.

        Equação:
        --------
        z = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = w₀ + Σ(wᵢxᵢ) para i=1 até n

        Em notação vetorial: z = X · w
        onde X inclui o termo de bias (x₀ = 1)

        Args:
            X: Matriz de features (com bias)

        Returns:
            Soma ponderada para cada amostra
        """
        return np.dot(X, self.weights_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """
        Treina o algoritmo.

        Pseudocódigo:
        -------------
        Para cada época t = 1, 2, ..., max_epochs:
          Para cada amostra i:
            1. Calcular net input: z⁽ⁱ⁾ = w₀ + Σ(wⱼx⁽ⁱ⁾ⱼ) para j=1 até n
            2. Calcular predição: ŷ⁽ⁱ⁾ = φ(z⁽ⁱ⁾)
            3. Calcular erro: e⁽ⁱ⁾ = y⁽ⁱ⁾ - ŷ⁽ⁱ⁾
            4. Atualizar pesos (Delta Rule):
               wⱼ⁽ᵗ⁺¹⁾ = wⱼ⁽ᵗ⁾ + η · e⁽ⁱ⁾ · x⁽ⁱ⁾ⱼ para todo j

        Critério de Parada:
        -------------------
        - Máximo de épocas atingido, ou
        - Convergência: nenhum erro em uma época completa (e⁽ⁱ⁾ = 0 para todo i)

        Args:
            X: Matriz de features com shape (n_samples, n_features)
            y: Vetor de labels com shape (n_samples,) contendo 0s e 1s

        Returns:
            self: Instância treinada do Perceptron

        Raises:
            ValueError: Se os dados de entrada são inválidos
        """
        # Validação dos dados de entrada
        X = np.array(X)
        y = np.array(y)

        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")
        if y.ndim != 1:
            raise ValueError("y deve ser um vetor 1D")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y devem ter o mesmo número de amostras")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("y deve conter apenas valores 0 e 1")

        # Inicializa pesos
        self._initialize_weights(X.shape[1])

        # Adiciona bias aos dados
        X_with_bias = self._add_bias(X)

        # Reinicia contadores
        self.errors_ = []
        self.n_epochs_ = 0

        # Loop de treinamento
        for epoch in range(self.max_epochs):
            errors = 0

            # Itera sobre todas as amostras
            for i in range(X.shape[0]):
                # Calcula predição
                net_input = self._net_input(X_with_bias[i])
                prediction = self._activation_function(net_input)

                # Calcula erro
                error = y[i] - prediction

                # Atualiza pesos se houver erro
                if error != 0:
                    self.weights_ += self.learning_rate * error * X_with_bias[i]
                    errors += 1

            # Armazena número de erros desta época
            self.errors_.append(errors)
            self.n_epochs_ = epoch + 1

            # Critério de parada: convergência (sem erros)
            if errors == 0:
                break

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Realiza predições para novos dados.

        Args:
            X: Matriz de features com shape (n_samples, n_features)

        Returns:
            Predições binárias (0 ou 1) com shape (n_samples,)

        Raises:
            ValueError: Se o modelo não foi treinado ou dados são inválidos
        """
        if not self.is_fitted_:
            raise ValueError("Modelo deve ser treinado antes de fazer predições")

        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        # Adiciona bias e calcula predições
        X_with_bias = self._add_bias(X)
        net_input = self._net_input(X_with_bias)
        return self._activation_function(net_input)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Calcula a soma ponderada para as amostras.

        Nota: O Perceptron clássico não produz probabilidades reais,
        mas retorna a soma ponderada que pode ser interpretada como
        uma medida de confiança.

        Args:
            X: Matriz de features com shape (n_samples, n_features)

        Returns:
            Soma ponderada para cada amostra
        """
        if not self.is_fitted_:
            raise ValueError(
                "Modelo deve ser treinado antes de calcular probabilidades"
            )

        X = np.array(X)
        if X.ndim != 2:
            raise ValueError("X deve ser uma matriz 2D")

        X_with_bias = self._add_bias(X)
        return self._net_input(X_with_bias)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calcula a acurácia do modelo nos dados fornecidos.

        Args:
            X: Matriz de features
            y: Vetor de labels verdadeiros

        Returns:
            Acurácia (proporção de predições corretas)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_decision_boundary(
        self, x_range: Tuple[float, float]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcula pontos da fronteira de decisão para dados 2D.

        Equação:
        --------
        A fronteira de decisão é o conjunto de pontos onde:
        w₀ + w₁x₁ + w₂x₂ = 0

        Resolvendo para x₂:
        x₂ = -(w₀ + w₁x₁) / w₂

        Esta linha separa as duas classes no espaço de features.
        Pontos acima da linha: classe 1 (z > 0)
        Pontos abaixo da linha: classe 0 (z < 0)

        Args:
            x_range: Tupla com (x_min, x_max) para o range do eixo x

        Returns:
            Tupla com (x_points, y_points) da linha de decisão

        Raises:
            ValueError: Se o modelo não foi treinado para dados 2D
        """
        if not self.is_fitted_:
            raise ValueError(
                "Modelo deve ser treinado antes de calcular fronteira de decisão"
            )

        if len(self.weights_) != 3:  # bias + 2 features
            raise ValueError("Fronteira de decisão só pode ser calculada para dados 2D")

        x_min, x_max = x_range
        x_points = np.linspace(x_min, x_max, 100)

        # Fronteira de decisão: w0 + w1*x1 + w2*x2 = 0
        # Resolvendo para x2: x2 = -(w0 + w1*x1) / w2
        w0, w1, w2 = self.weights_

        if w2 == 0:
            # Linha vertical quando w2 = 0
            y_points = np.full_like(x_points, -w0 / w1 if w1 != 0 else 0)
        else:
            y_points = -(w0 + w1 * x_points) / w2

        return x_points, y_points

    def get_params(self) -> dict:
        """
        Retorna os parâmetros do modelo.

        Returns:
            Dicionário com os parâmetros do modelo
        """
        return {
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "random_state": self.random_state,
            "weights": self.weights_.tolist() if self.weights_ is not None else None,
            "n_epochs": self.n_epochs_,
            "is_fitted": self.is_fitted_,
        }

    def __repr__(self) -> str:
        """Representação do objeto."""
        return (
            f"Perceptron(learning_rate={self.learning_rate}, "
            f"max_epochs={self.max_epochs}, "
            f"random_state={self.random_state})"
        )
