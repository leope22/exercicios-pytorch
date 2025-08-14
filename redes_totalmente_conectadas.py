import torch
import random
from pi import Solucionador


class ConjuntoDeDados(torch.utils.data.Dataset):

    def __init__(self, X, y):
        """
        Inicializa o conjunto de dados.
        Entrada:
        - X: Tensor de dados de entrada de shape (N, d_1, ..., d_k)
        - y: int64 Tensor de rótulos, e shape (N,). y[i] fornece o rótulo para X[i].
        """
        self.X = None
        self.y = None

        self.X = X
        self.y = y

    def __getitem__(self, i):
        """
        Retorna a i-esima amostra do conjunto de dados.
        Entrada:
        - i: inteiro indicando o número da amostra desejada
        Retorno:
        - amostra: tupla contendo os dados e o rótulo da i-ésima amostra
        """
        amostra = None

        amostra = (self.X[i], self.y[i])

        return amostra

    def __len__(self):
        """
        Retorna o número total de amostras no conjunto de dados.
        Retorno:
        - num_amostras: inteiro indicando o número total de amostras.
        """
        num_amostras = len(self.X)

        return num_amostras


class RedeDuasCamadas(torch.nn.Module):
    """
    Uma rede neural totalmente conectada de duas camadas com não linearidade ReLU.
    Assumimos uma dimensão de entrada de D, uma dimensão oculta de H, e realizar a
    classificação em C classes. A arquitetura deve ser linear - relu - linear.
    Observe que esta classe não implementa a decida de gradiente; em vez disso, ela
    irá interagir com um objeto separado que é responsável por executar a otimização.
    """

    def __init__(
        self,
        dim_entrada=3 * 32 * 32,
        dim_oculta=100,
        num_classes=10,
        escala_peso=1e-3,
        dtype=torch.float32,
        device="cpu",
    ):
        """
        Inicializa o modelo. Pesos são inicializados com pequenos valores aleatórios
        e vieses são inicializados com zero.

        W1: Pesos da primeira camada; tem shape (D, H)
        b1: Vieses da primeira camada; tem shape (H,)
        W2: Pesos da segunda camada; tem shape (H, C)
        b2: Vieses da segunda camada; tem shape (C,)

        Entrada:
        - dim_entrada: A dimensão D dos dados de entrada.
        - dim_oculta: O número de neurônios H na camada oculta.
        - num_classes: O número de categorias C.
        - escala_peso: Escalar indicando o desvio padrão para inicialização dos pesos
        - dtype: Opcional, tipo de dados de cada parâmetro de peso.
        - device: Opcional, se os parâmetros de peso estão na GPU ou CPU.
        """
        super().__init__()

        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

        self.W1 = torch.nn.Parameter(
            escala_peso
            * torch.randn(dim_entrada, dim_oculta, dtype=dtype, device=device)
        )
        self.b1 = torch.nn.Parameter(
            torch.zeros(dim_oculta, dtype=dtype, device=device)
        )
        self.W2 = torch.nn.Parameter(
            escala_peso
            * torch.randn(dim_oculta, num_classes, dtype=dtype, device=device)
        )
        self.b2 = torch.nn.Parameter(
            torch.zeros(num_classes, dtype=dtype, device=device)
        )

    def forward(self, X):
        """
        Executa o passo para frente da rede para calcular as pontuações de classe.
        A arquitetura da rede deve ser:

        camada linear -> ReLU (rep_oculta) -> camada linear (pontuações)

        Entrada:
        - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

        Retorno:
        - pontuacoes: Tensor de shape (N, C) contendo pontuações de classe, em
          que pontuacoes[i, c] é a pontuação de categoria para X[i] e classe c.
        """
        pontuacoes = None

        h1 = X.mm(self.W1) + self.b1
        h1_relu = torch.relu(h1)
        pontuacoes = h1_relu.mm(self.W2) + self.b2

        return pontuacoes


def perda_softmax(x, y):
    """
    Calcula a perda usando a classificação softmax.

    Entrada:
    - x: Dados de entrada, de shape (N, C), onde x[i, j] é a pontuação para a
      j-ésima classe para a i-ésima amostra de treinamento.
    - y: Vetor de rótulos, de shape (N,), onde y[i] é o rótulo para x[i] e
      0 <= y[i] < C.

    Retorno:
    - perda: escalar fornecendo a perda
    """
    perda = None

    perda = torch.nn.functional.cross_entropy(x, y.to(x.device))

    return perda


class RedeTotalmenteConectada(torch.nn.Module):
    """
    Uma rede neural totalmente conectada com um número arbitrário de camadas
    ocultas e ativações ReLU. Para uma rede com camadas L, a arquitetura será:

    {linear - relu - [descarte]} x (L - 1) - linear

    onde descarte é opcional, e o bloco {...} é repetido L - 1 vezes.
    """

    def __init__(
        self,
        dims_oculta,
        dim_entrada=3 * 32 * 32,
        num_classes=10,
        descarte=0.0,
        escala_peso=1e-2,
    ):
        """
        Inicialize uma nova RedeTotalmenteConectada.

        Entradas:
        - dims_oculta: uma lista de inteiros indicando o tamanho de cada camada oculta.
        - dim_entrada: um inteiro indicando o tamanho da entrada.
        - num_classes: Um inteiro indicando o número de categorias a serem classificadas.
        - descarte: escalar entre 0 e 1 indicando a probabilidade de descarte para redes
          com camadas de descarte. Se descarte = 0, então a rede não deve usar descarte.
        - escala_peso: Escalar indicando o desvio padrão para inicialização dos pesos
        """
        super().__init__()

        random.seed(0)
        torch.manual_seed(0)

        self.descarte = descarte
        self.usar_descarte = descarte != 0
        self.escala_peso = escala_peso
        self.num_camadas = 1 + len(dims_oculta)

        self.camadas = torch.nn.ModuleList()
        dims = [dim_entrada] + dims_oculta
        for i in range(self.num_camadas - 1):
            self.camadas.append(torch.nn.Linear(dims[i], dims[i + 1]))
        self.camadas.append(torch.nn.Linear(dims_oculta[-1], num_classes))

        self.reset_parameters()

    def forward(self, X):
        """
        Executa o passo para frente da rede para calcular as pontuações de classe.

        Entrada:
        - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

        Retorno:
        - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
        """
        pontuacoes = None
        h = X
        for i in range(self.num_camadas - 1):
            h = self.camadas[i](h)
            h = torch.relu(h)
            if self.usar_descarte:
                h = torch.nn.functional.dropout(
                    h, p=self.descarte, training=self.training
                )
        pontuacoes = self.camadas[-1](h)

        return pontuacoes

    def reset_parameters(self):
        """
        Inicializa os pesos e vieses das camadas totalmente conectadas.
        """
        for nome, camada in self.named_modules():
            if isinstance(camada, torch.nn.Linear):
                torch.nn.init.normal_(camada.weight, std=self.escala_peso)
                torch.nn.init.zeros_(camada.bias)


def criar_instancia_solucionador(dic_dados, dtype, device):
    modelo = RedeDuasCamadas(dim_oculta=512, dtype=dtype, device=device)
    solucionador = None
    dados = {
        "treinamento": ConjuntoDeDados(dic_dados["X_treino"], dic_dados["y_treino"]),
        "validacao": ConjuntoDeDados(dic_dados["X_val"], dic_dados["y_val"]),
    }
    taxa_aprendizagem = 1e-2
    reg = 1e-4
    num_epocas = 20
    tamanho_lote = 256
    funcao_de_perda = perda_softmax
    otimizador = torch.optim.SGD(
        modelo.parameters(), lr=taxa_aprendizagem, momentum=0.9, weight_decay=reg
    )
    solucionador = Solucionador(
        modelo,
        funcao_de_perda,
        otimizador,
        dados,
        num_epocas=num_epocas,
        tamanho_lote=tamanho_lote,
        device=device,
        verbose=True,
    )
    return solucionador


def retorna_params_rede_tres_camadas():
    taxa_aprendizagem = 1e-1
    escala_peso = 1e-1

    return escala_peso, taxa_aprendizagem


def retorna_params_rede_cinco_camadas():
    taxa_aprendizagem = 1e-1
    escala_peso = 1e-1

    return escala_peso, taxa_aprendizagem
