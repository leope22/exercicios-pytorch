# Exercícios de PyTorch: Do Básico a Redes Convolucionais

Este repositório contém uma coleção de códigos em Python que demonstram a implementação de conceitos fundamentais de PyTorch, desde a manipulação básica de tensores até a construção e treinamento de classificadores lineares, redes neurais totalmente conectadas (FC) e redes neurais convolucionais (CNNs).

## Estrutura do Repositório

O projeto é composto por quatro arquivos principais, cada um focado em um conjunto específico de habilidades e conceitos:

1.  `pytorch101.py`
    * **Descrição:** Funções para a prática de operações essenciais com tensores no PyTorch.
    * **Conceitos:** Criação, indexação, fatiamento (`slicing`), remodelagem (`reshaping`), operações de redução e uso de GPU para multiplicação de matrizes.

2.  `classificador_linear.py`
    * **Descrição:** Implementação de classificadores lineares (SVM e Softmax) a partir do zero.
    * **Conceitos:** Funções de perda (ingênua com laços e vetorizada), cálculo de gradiente, treinamento com Descida de Gradiente Estocástico (SGD) e busca por hiperparâmetros.

3.  `redes_totalmente_conectadas.py`
    * **Descrição:** Construção de redes neurais totalmente conectadas (`fully-connected`) utilizando o módulo `torch.nn`.
    * **Conceitos:** Definição de arquiteturas de múltiplas camadas (duas camadas e L-camadas), uso de `torch.nn.Module`, função de perda `cross-entropy`, e treinamento de modelos com um `Solucionador` genérico.

4.  `redes_convolucionais.py`
    * **Descrição:** Implementação de redes neurais convolucionais (CNNs) para tarefas de classificação de imagem.
    * **Conceitos:** Construção de camadas convolucionais (`Conv2d`), de `pooling` (`MaxPool2d`) e de normalização em lote (`BatchNorm2d`). Demonstração de uma CNN simples e de uma arquitetura profunda inspirada na VGG.

## Conceitos Abordados

-   **Fundamentos do PyTorch:** Manipulação de tensores, operações matemáticas e broadcasting.
-   **Indexação Avançada:** Fatiamento, indexação com arrays de inteiros e máscaras booleanas.
-   **Autograd:** Cálculo automático de gradientes.
-   **Machine Learning:** Implementação de funções de perda (SVM, Softmax com Entropia Cruzada) e algoritmos de otimização.
-   **Redes Neurais:**
    -   Construção de modelos com `torch.nn.Module`.
    -   Implementação de Redes Totalmente Conectadas (FCNs).
    -   Implementação de Redes Convolucionais (CNNs).
    -   Técnicas como Ativação ReLU, Dropout e Normalização em Lote.
-   **Treinamento de Modelos:** Ciclos de treinamento, processamento em lote (`batch processing`), otimizadores (SGD, Adam) e avaliação de acurácia.
-   **Computação em GPU:** Aceleração de cálculos movendo tensores para a GPU (`.cuda()`).

Este repositório é destinado a fins de estudo e prática. Cada arquivo contém um conjunto de funções que podem ser executadas individualmente em um ambiente Python com a biblioteca PyTorch instalada.
