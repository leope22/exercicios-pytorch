import torch
import random
import statistics
from abc import abstractmethod

class ClassificadorLinear(object):
  """ Uma classe abstrata para os classificadores lineares """
  def __init__(self):
    random.seed(0)
    torch.manual_seed(0)
    self.W = None

  def treinar(self, X_treino, y_treino, taxa_aprendizagem=1e-3, reg=1e-5, num_iters=100,
            tamanho_lote=200, verbose=False):
    args_treino = (self.perda, self.W, X_treino, y_treino, taxa_aprendizagem, reg,
                   num_iters, tamanho_lote, verbose)
    self.W, historico_perda = treinar_classificador_linear(*args_treino)
    return historico_perda

  def prever(self, X):
    return prever_classificador_linear(self.W, X)

  @abstractmethod
  def perda(self, W, X_lote, y_lote, reg):
    """
    Calcula a função de perda e a sua derivada.
    As subclasses irão sobrescrever isso.

    Entrada:
    - W: Um tensor PyTorch de shape (D, C) contendo pesos (treinados) de um modelo.
    - X_lote: Um tensor PyTorch de shape (N, D) contendo um minilote de N
      amostras de dados; cada amostra tem dimensão D.
    - y_lote: Um tensor PyTorch de shape (N,) contendo rótulos para o minilote.
    - reg: (float) força de regularização.

    Retorno: Uma tupla contendo:
    - perda como um único float
    - gradiente em relação a self.W; um tensor de mesmo shape de W
    """
    raise NotImplementedError

  def _perda(self, X_lote, y_lote, reg):
    self.perda(self.W, X_lote, y_lote, reg)

  def salvar(self, path):
    torch.save({'W': self.W}, path)
    print("Salvo em {}".format(path))

  def carregar(self, path):
    W_dict = torch.load(path, map_location='cpu')
    self.W = W_dict['W']
    print("Carregando arquivo de ponto de verificação: {}".format(path))

class SVMLinear(ClassificadorLinear):
  """ Uma subclasse que usa a função de perda SVM multiclasse """
  def perda(self, W, X_lote, y_lote, reg):
    return perda_svm_vetorizada(W, X_lote, y_lote, reg)

class Softmax(ClassificadorLinear):
  """ Uma subclasse que usa a função de perda Softmax + Entropia-cruzada """
  def perda(self, W, X_lote, y_lote, reg):
    return perda_softmax_vetorizada(W, X_lote, y_lote, reg)

def perda_svm_ingenua(W, X, y, reg):
  """
  Função de perda de SVM estruturada, implementação ingênua (com laços).

  As entradas têm dimensão D, existem C classes e operamos em minilotes
  de N amostras. Ao implementar a regularização em W, por favor, NÃO
  multiplique o termo de regularização por 1/2 (sem coeficiente).

  Entrada:
  - W: Um tensor PyTorch de shape (D, C) contendo pesos.
  - X: Um tensor PyTorch de shape (N, D) contendo um minilote de dados.
  - y: Um tensor PyTorch de shape (N,) contendo rótulos de treino; y[i] = c 
    significa que X[i] tem rótulo c, onde 0 <= c < C.
  - reg: (float) força de regularização

  Retorno: Uma tupla de:
  - perda como escalar do torch
  - gradiente da perda em relação aos pesos W; um tensor de mesmo shape de W    
  """
  dW = torch.zeros_like(W)
  num_classes = W.shape[1]
  num_treino = X.shape[0]
  perda = 0.0
  for i in range(num_treino):
    pontuacoes = W.t().mv(X[i])
    pontuacao_classe_correta = pontuacoes[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margem = pontuacoes[j] - pontuacao_classe_correta + 1
      if margem > 0:
        perda += margem
        dW[:, j] += X[i] / num_treino
        dW[:, y[i]] -= X[i] / num_treino

  perda /= num_treino
  perda += reg * torch.sum(W * W)
  dW += 2 * reg * W

  return perda, dW

def perda_svm_vetorizada(W, X, y, reg):
  """
  Função de perda de SVM estruturada, implementação vetorizada. Ao implementar 
  a regularização em W, por favor, NÃO multiplique o termo de regularização por 
  1/2 (sem coeficiente). As entradas e saídas são as mesmas de perda_svm_ingenua.
        
  Entrada:
  - W: Um tensor PyTorch de shape (D, C) contendo pesos.
  - X: Um tensor PyTorch de shape (N, D) contendo um minilote de dados.
  - y: Um tensor PyTorch de shape (N,) contendo rótulos de treino; y[i] = c 
    significa que X[i] tem rótulo c, onde 0 <= c < C.
  - reg: (float) força de regularização

  Retorno: Uma tupla de:
  - perda como escalar do torch
  - gradiente da perda em relação aos pesos W; um tensor de mesmo shape de W    
  """
  perda = 0.0
  dW = torch.zeros_like(W)
  num_treino = X.shape[0]
  pontuacoes = X.mm(W)
  pontuacoes_corretas = pontuacoes[torch.arange(num_treino), y].view(-1, 1)
  margens = pontuacoes - pontuacoes_corretas + 1
  margens[torch.arange(num_treino), y] = 0
  margens = torch.clamp(margens, min=0)
  perda = torch.sum(margens) / num_treino + reg * torch.sum(W * W)
  mascara = (margens > 0).float().type_as(X)
  mascara[torch.arange(num_treino), y] = -torch.sum(mascara, dim=1)
  dW = X.t().mm(mascara) / num_treino + 2 * reg * W

  return perda, dW

def amostrar_lote(X, y, num_treino, tamanho_lote):
  """
  Amostra tamanho_lote elementos dos dados de treino e seus rótulos 
  correspondentes para usar nesta rodada de descida de gradiente.
  """
  X_lote = None
  y_lote = None

  indices = torch.randint(0, num_treino, (tamanho_lote,), device=X.device)
  X_lote = X[indices]
  y_lote = y[indices]

  return X_lote, y_lote

def treinar_classificador_linear(funcao_perda, W, X, y, taxa_aprendizagem=1e-3,
                                 reg=1e-5, num_iters=100, tamanho_lote=200,
                                 verbose=False):
  """
  Treina este classificador linear usando a descida do gradiente estocástico.

  Entrada:
  - funcao_perda: função de perda para usar durante o treino. Deve receber W, 
    X, y e reg como entrada e produzir como saída de uma tupla de (perda, dW)
  - W: Um tensor PyTorch de shape (D, C) fornecendo os pesos iniciais do
    classificador. Se W for None, ele será inicializado aqui.
  - X: Um tensor PyTorch de shape (N, D) contendo dados de treino; há N
    amostras de treino cada uma com D dimensões.
  - y: Um tensor PyTorch de shape (N,) contendo rótulos de treino; y[i] = c
    significa que X[i] tem rótulo 0 <= c < C para C classes.
  - taxa_aprendizagem: (float) taxa de aprendizagem para otimização.
  - reg: (float) força de regularização.
  - num_iters: (inteiro) número de passos a serem executadas ao otimizar
  - tamanho_lote: (inteiro) número de amostras de treino a serem usados ​​em cada passo.
  - verbose: (booleano) Se verdadeiro, imprime o progresso durante a otimização.

  Retorno: uma tupla de:
  - W: O valor final da matriz de pesos e o final da otimização
  - historico_perda: uma lista de escalares Python fornecendo os valores da perda 
    em cada iteração de treino.
  """
  num_treino, dim = X.shape
  if W is None:
    num_classes = torch.max(y) + 1
    W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)
  else:
    num_classes = W.shape[1]

  historico_perda = []
  for it in range(num_iters):
    X_lote, y_lote = amostrar_lote(X, y, num_treino, tamanho_lote)

    perda, grad = funcao_perda(W, X_lote, y_lote, reg)
    historico_perda.append(perda.item())

    W -= taxa_aprendizagem * grad

    if verbose and it % 100 == 0:
      print('iteração %d / %d: perda %f' % (it, num_iters, perda))

  return W, historico_perda


def prever_classificador_linear(W, X):
  """
  Usa os pesos treinados deste classificador linear para prever rótulos 
  para pontos de dados.    

  Entrada:
  - W: Um tensor PyTorch de shape (D, C), contendo pesos de um modelo
  - X: Um tensor PyTorch de shape (N, D) contendo dados de treino; há N
    amostras de treino cada uma com D dimensões.

  Retorno:
  - y_prev: Um tensor PyTorch de dtype int64 e shape (N,) fornecendo rótulos previstos 
    para cada elemento de X. Cada elemento de y_prev deve estar entre 0 e C - 1.
  """
  y_prev = torch.zeros(X.shape[0], dtype=torch.int64)
  scores = X.mm(W)
  y_prev = torch.argmax(scores, dim=1)

  return y_prev


def svm_retorna_params_busca():
  """
  Retorna hiperparâmetros candidatos para o modelo SVM. Você deve fornecer 
  pelo menos dois parâmetros para cada um, e o total de combinações de busca 
  em grade deve ser inferior a 25.

  Retorno:
  - taxas_aprendizagem: candidatos a taxa de aprendizagem, p.ex. [1e-3, 1e-2, ...]
  - forcas_regularizacao: candidatos a força de regularização 
                          p.ex. [1e0, 1e1, ...]
  """
  taxas_aprendizagem = []
  forcas_regularizacao = []

  taxas_aprendizagem = [5.5e-7, 3.5e-7, 2.5e-7]
  forcas_regularizacao = [750.0, 1250.0, 1750.0]

  return taxas_aprendizagem, forcas_regularizacao


def testar_combinacao_params(cls, dic_dados, tx, reg, num_iters=2000):
  """
  Treina uma única instância ClassificadorLinear e retorna a instância aprendida 
  com acurácia de treino/val.

  Entrada:
  - cls (ClassificadorLinear): uma instância ClassificadorLinear recém-criada.
                               O treinamento/validação deve ser executado nesta instância
  - dic_dados (dict): um dicionário que inclui
                      ['X_treino', 'y_treino', 'X_val', 'y_val']
                      como as chaves para treinar um classificador
  - lr (float): uma taxa de aprendizagem para treinar uma instância SVM.
  - reg (float): um peso de regularização para treinar uma instância SVM.
  - num_iters (int, opcional): um número de iterações para treinar

  Retorno:
  - cls (ClassificadorLinear): uma instância de ClassificadorLinear treinada com
                               (['X_treino', 'y_treino'], lr, reg)
                               por num_iter vezes.
  - acc_treino (float): acurácia de treinamento do modelo SVM
  - acc_val (float): acurácia de validação do modelo SVM
  """
  acc_treino = 0.0
  acc_val = 0.0
  #num_iters = 100

  cls.treinar(dic_dados['X_treino'], dic_dados['y_treino'], taxa_aprendizagem=tx, reg=reg, num_iters=num_iters)
  
  y_pred_treino = cls.prever(dic_dados['X_treino'])
  acc_treino = (y_pred_treino == dic_dados['y_treino']).float().mean().item()
  
  y_pred_val = cls.prever(dic_dados['X_val'])
  acc_val = (y_pred_val == dic_dados['y_val']).float().mean().item()

  return cls, acc_treino, acc_val

def perda_softmax_ingenua(W, X, y, reg):
  """
  Função de perda de Softmax, implementação ingênua (com laços). Ao implementar 
  a regularização sobre W, NÃO multiplique o termo de regularização por 
  1/2 (sem coeficiente).

  As entradas têm dimensão D, existem C classes e operamos em minilotes 
  de N amostras.

  Entrada:
  - W: Um tensor PyTorch de shape (D, C) contendo pesos.
  - X: Um tensor PyTorch de shape (N, D) contendo um minilote de dados.
  - y: Um tensor PyTorch de shape (N,) contendo rótulos de treinamento; y[i] = c 
    significa que X[i] tem rótulo c, onde 0 <= c < C.
  - reg: (float) força de regularização

  Retorno: Uma tupla de:
  - perda como um único float
  - gradiente em relação aos pesos W; um tensor de mesmo shape de W
  """
  perda = 0.0
  dW = torch.zeros_like(W)

  N = X.shape[0]
  C = W.shape[1]

  for i in range(N):
    scores = X[i] @ W
    scores -= torch.max(scores)
    exp_scores = torch.exp(scores)
    probs = exp_scores / torch.sum(exp_scores)
    
    perda += -torch.log(probs[y[i]])

    for j in range(C):
      if j == y[i]:
        dW[:, j] += (probs[j] - 1) * X[i]
      else:
        dW[:, j] += probs[j] * X[i]

  perda /= N
  dW /= N

  perda += reg * torch.sum(W * W)
  dW += 2 * reg * W

  return perda, dW


def perda_softmax_vetorizada(W, X, y, reg):
  """
  Função de perda Softmax, versão vetorizada. Ao implementar a regularização 
  sobre W, NÃO multiplique o termo de regularização por 1/2 (sem coeficiente).

  As entradas e saídas são iguais a perda_softmax_ingenua.    
  """
  perda = 0.0
  dW = torch.zeros_like(W)

  num_treino = X.shape[0]
  
  scores = X.mm(W)
  
  scores -= scores.max(dim=1, keepdim=True).values
  
  exp_scores = torch.exp(scores)
  probs = exp_scores / exp_scores.sum(dim=1, keepdim=True)
  
  correct_logprobs = -torch.log(probs[range(num_treino), y])
  perda = correct_logprobs.mean()
  
  perda += reg * torch.sum(W * W)
  
  dscores = probs.clone()
  dscores[range(num_treino), y] -= 1
  dscores /= num_treino
  dW = X.t().mm(dscores)
  
  dW += 2 * reg * W

  return perda, dW


def softmax_retorna_params_busca():
  """
  Retorna hiperparâmetros candidatos para o modelo Softmax. Você deve fornecer 
  pelo menos dois parâmetros para cada um, e o total de combinações de busca 
  em grade deve ser inferior a 25.

  Retorno:
  - taxas_aprendizagem: candidatos a taxa de aprendizagem, p.ex. [1e-3, 1e-2, ...]
  - forcas_regularizacao: candidatos a força de regularização 
                          p.ex. [1e0, 1e1, ...]
  """
  taxas_aprendizagem = []
  forcas_regularizacao = []

  taxas_aprendizagem = [5e-4, 1e-3, 5e-3, 1e-2]
  forcas_regularizacao = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]

  return taxas_aprendizagem, forcas_regularizacao