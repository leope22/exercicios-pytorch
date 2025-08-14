import torch
import random
from pi import Solucionador
from redes_totalmente_conectadas import *

class RedeConvTresCamadas(torch.nn.Module):
  """
  Uma rede convolucional de três camadas com a seguinte arquitetura:
  conv - relu - max pooling 2x2 - linear - relu - linear
  A rede opera em mini-lotes de dados que têm shape (N, C, H, W)
  consistindo em N imagens, cada uma com altura H e largura W e com C
  canais de entrada.
  """

  def __init__(self, dims_entrada=(3, 32, 32), num_filtros=32, tamanho_filtro=7,
               dim_oculta=100, num_classes=10, escala_peso=1e-3):
    """
    Inicializa a nova rede.
    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Número de filtros a serem usados na camada de convolução
    - tamanho_filtro: Largura/altura dos filtros a serem usados na camada de convolução
    - dim_oculta: Número de unidades a serem usadas na camada oculta totalmente conectada
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos.
    """
    super().__init__()

    random.seed(0)
    torch.manual_seed(0)
    
    self.escala_peso = escala_peso
    C, H, W = dims_entrada

    self.conv = torch.nn.Conv2d(
        in_channels=C,
        out_channels=num_filtros,
        kernel_size=tamanho_filtro,
        padding=(tamanho_filtro-1)//2
    )
    
    self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    dim_apos_pool = num_filtros * (H // 2) * (W // 2)
    self.fc1 = torch.nn.Linear(dim_apos_pool, dim_oculta)
    self.fc2 = torch.nn.Linear(dim_oculta, num_classes)
    self.relu = torch.nn.ReLU()

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

    out = self.relu(self.conv(X))
    out = self.pool(out)
    out = out.view(out.size(0), -1)
    out = self.relu(self.fc1(out))
    pontuacoes = self.fc2(out)
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for param in self.parameters():
      if isinstance(param, torch.nn.Conv2d) or isinstance(param, torch.nn.Linear):
        torch.nn.init.normal_(param.weight, std=self.escala_peso)
        torch.nn.init.zeros_(param.bias)


class RedeConvProfunda(torch.nn.Module):
  """
  Uma rede neural convolucional com um número arbitrário de camadas de 
  convolução no estilo da rede VGG. Todas as camadas de convolução usarão 
  filtro de tamanho 3 e preenchimento de 1 para preservar o tamanho do mapa 
  de ativação, e todas as camadas de agrupamento serão camadas de agrupamento 
  por máximo com campos receptivos de 2x2 e um passo de 2 para reduzir pela 
  metade o tamanho do mapa de ativação.

  A rede terá a seguinte arquitetura:

  {conv - [normlote?] - relu - [agrup?]} x (L - 1) - linear

  Cada estrutura {...} é uma "camada macro" que consiste em uma camada de 
  convolução, uma camada de normalização de lote opcional, uma não linearidade 
  ReLU e uma camada de agrupamento opcional. Depois de L-1 dessas macrocamadas, 
  uma única camada totalmente conectada é usada para prever pontuações de classe.

  A rede opera em minilotes de dados que possuem shape (N, C, H, W) consistindo 
  de N imagens, cada uma com altura H e largura W e com C canais de entrada.
  """
  def __init__(self, dims_entrada=(3, 32, 32),
               num_filtros=[8, 8, 8, 8, 8],
               agrups_max=[0, 1, 2, 3, 4],
               normlote=False,
               num_classes=10, escala_peso=1e-3):
    """
    Inicializa uma nova rede.

    Entrada:
    - dims_entrada: Tupla (C, H, W) indicando o tamanho dos dados de entrada
    - num_filtros: Lista de comprimento (L - 1) contendo o número de filtros
      de convolução para usar em cada macrocamada.
    - agrups_max: Lista de inteiros contendo os índices (começando em zero) das 
      macrocamadas que devem ter agrupamento por máximo.
    - normlote: Booleano dizendo se normalização do lote deve ou não ser 
      incluída em cada macrocamada.
    - num_classes: Número de pontuações a serem produzidas na camada linear final.
    - escala_peso: Escalar indicando o desvio padrão para inicialização 
      aleatória de pesos, ou a string "kaiming" para usar a inicialização Kaiming.
    """
    super().__init__()

    random.seed(0)
    torch.manual_seed(0)
    
    self.num_camadas = len(num_filtros)+1
    self.escala_peso = escala_peso
    self.agrups_max = agrups_max
    self.normlote = normlote

    C, H, W = dims_entrada
    self.camadas = torch.nn.ModuleList()
    in_channels = C
    
    for i, out_channels in enumerate(num_filtros):
        conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.camadas.append(conv)
        
        if normlote:
            bn = torch.nn.BatchNorm2d(out_channels)
            self.camadas.append(bn)
            torch.nn.init.ones_(bn.weight)
            torch.nn.init.zeros_(bn.bias)
        
        self.camadas.append(torch.nn.ReLU())
        
        if i in agrups_max:
            pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
            self.camadas.append(pool)
            H = H // 2
            W = W // 2
        
        in_channels = out_channels
    
    self.linear = torch.nn.Linear(in_channels * H * W, num_classes)

    if not self.normlote:
      params_por_camada_macro = 2
    else:
      params_por_camada_macro = 4
    num_params = params_por_camada_macro * len(num_filtros) + 2
    msg = 'self.parameters() tem o número errado de ' \
          'elementos. Obteve %d; esperava %d'
    msg = msg % (len(list(self.parameters())), num_params)
    assert len(list(self.parameters())) == num_params, msg
    
    self.reset_parameters()

  def forward(self, X):
    """
    Executa o passo para frente da rede para calcular as pontuações de classe.

    Entrada:
    - X: Dados de entrada de shape (N, D). Cada X[i] é uma amostra de treinamento.

    Retorno: 
    - pontuacoes: Tensor de shape (N, C) contendo as pontuações de classe para X
    """
    out = X
    for camada in self.camadas:
        out = camada(out)
    out = out.view(out.size(0), -1)
    pontuacoes = self.linear(out)  
    
    return pontuacoes
    
  def reset_parameters(self):
    """
    Inicializa os pesos e vieses das camadas convolucionais e totalmente conectadas.
    """
    for nome, camada in self.named_modules():
      if isinstance(camada, torch.nn.Conv2d) or isinstance(camada, torch.nn.Linear):
        if isinstance(self.escala_peso, str) and self.escala_peso == "kaiming":
          torch.nn.init.kaiming_normal_(camada.weight, mode='fan_out', nonlinearity='relu')       
        else:
          torch.nn.init.normal_(camada.weight, std=self.escala_peso)
        torch.nn.init.zeros_(camada.bias)


def encontrar_parametros_sobreajuste():
  taxa_aprendizagem = 5e-3
  escala_peso = 5e-2

  return escala_peso, taxa_aprendizagem


def criar_instancia_solucionador_convolucional(dic_dados, device):
  modelo = None
  solucionador = None

  dims_entrada = dic_dados['X_treino'].shape[1:]
  num_classes = 10
  
  modelo = RedeConvProfunda(dims_entrada=dims_entrada, num_classes=num_classes, num_filtros=[32, 64, 128, 256, 256], agrups_max=[1, 2, 3, 4], normlote=True, escala_peso='kaiming').to(device)
  otimizador = torch.optim.Adam(modelo.parameters(), lr=3e-4, weight_decay=1e-4)
  
  dados = {
      'treinamento': ConjuntoDeDados(dic_dados['X_treino'], dic_dados['y_treino']),
      'validacao': ConjuntoDeDados(dic_dados['X_val'], dic_dados['y_val'])
  }
  
  solucionador = Solucionador(modelo, perda_softmax, otimizador, dados, imprime_cada=100, num_epocas=20, tamanho_lote=128, device=device)
  return solucionador