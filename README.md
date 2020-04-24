# Classificação de dígitos SVM MNIST em python usando o scikit-learn

O projeto apresenta o conhecido problema da [classificação de dígitos manuscritos MNIST](https://en.wikipedia.org/wiki/MNIST_database).

Para os fins deste tutorial, usarei [Support Vector Machine (SVM)](https://en.wikipedia.org/wiki/Support_vector_machine) olhando pixels brutos.

A solução é escrita em python com o uso de [scikit-learn](http://scikit-learn.org/stable/) f

![Sample MNIST digits visualization](/images/mnist_digits.png)



O objetivo deste projeto não é alcançar o desempenho de ponta, mas sim ensinár ** como treinar o classificador SVM em dados de imagem ** com o uso do SVM do sklearn.

Embora a solução não seja otimizada para alta precisão, os resultados são bastante bons (consulte a tabela abaixo).

Se você deseja obter o melhor desempenho, esses dois recursos mostrarão as soluções atuais de última geração:

* [Quem é o melhor no MNIST?](Http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#4d4e495354)
* [Competição do reconhecedor de dígitos Kaggle](https://www.kaggle.com/c/digit-recognizer)

A tabela abaixo mostra alguns resultados em comparação com outros modelos:


| Method                                     | Accuracy | Comments     |
|--------------------------------------------|----------|--------------|
| Random forest                              | 0.937    |              |
| Simple one-layer neural network            | 0.926    |              |
| Simple 2 layer convolutional network       | 0.981    |              |
| SVM RBF                                    | 0.9852   | C=5, gamma=0.05 |
| Linear SVM + Nystroem kernel approximation |          |              |
| Linear SVM + Fourier kernel approximation  |          |              |


## Configuração do projeto

Este tutorial foi escrito e testado no Ubuntu 18.10.
Projeto contém o Pipfile com todas as bibliotecas necessárias

* Python - versão> = 3.6
* pipenv - gerenciamento de pacotes e ambiente virtual
* entorpecido
* matplotlib
* scikit-learn


1. Instale o Python.
1. [Instalar pipenv] (https://pipenv.readthedocs.io/en/latest/install/#pragmatic-installation-of-pipenv)
1. Git clona o repositório
1. Instale todos os pacotes python necessários executando este comando no terminal

```
git clone https://github.com/ksopyla/svm_mnist_digit_classification.git
cd svm_mnist_digit_classification
instalação do pipenv
```

## Solução

Neste tutorial, utilizo duas abordagens para o aprendizado do SVM.

Primeiro, usa o SVM clássico com o kernel RBF. A desvantagem desta solução é um treinamento bastante longo em grandes conjuntos de dados, embora a precisão com bons parâmetros seja alta.

O segundo, usa o SVM linear, que permite o treinamento em O (n) tempo. Para alcançar alta precisão, usamos alguns truques. Nós aproximamos o kernel RBF em um espaço de alta dimensão por incorporação. A teoria por trás é bastante complicada, no entanto [o sklearn está pronto para usar classes para aproximação do kernel] (http://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation).

Nós vamos usar:

* Aproximação do kernel Nystroem
* Aproximação do kernel de Fourier

O código foi testado com python 3.6.

## Como o projeto está organizado

Projeto consiste em três arquivos:

* _mnist_helpers.py_ - contém algumas funções de visualização: visualização de dígitos MNIST e matriz de confusão
* _svm_mnist_classification.py_ - script para SVM com classificação de kernel RBF
* _svm_mnist_embedings.py_ - script para SVM linear com incorporações

### SVM com kernel RBF

O script ** svm_mnist_classification.py ** baixa o banco de dados MNIST e visualiza alguns dígitos aleatórios.
Em seguida, padroniza os dados (média = 0, padrão = 1) e inicia a pesquisa na grade com validação cruzada para encontrar os melhores parâmetros.

1. Pesquisa de parâmetros RBF do kernel MNIST SVM C = [0.1,0.5,1,5], gama = [0,01,0.0.05,0.1,0.5].

A pesquisa em grade foi realizada para os parâmetros C e gama, onde C = [0,1,0,5,1,5], gama = [0,01,0.0.05,0.1,0.5].
Até agora, examinei apenas 4x4 pares de parâmetros diferentes com validação cruzada de três vezes (modelos 4x4x3 = 48), esse procedimento leva 3687,2 minutos (2 dias, 13: 56: 42.531223 exatamente) em uma CPU de núcleo.

O espaço param foi gerado com espaço de log numpy e multiplicação da matriz externa.
```
C_range = np.outer (np.logspace (-1, 0, 2), np.array ([1,5]))
# achatar matriz, mude para matriz numpy 1D
C_range = C_range.flatten ()

gamma_range = np.outer (np.logspace (-2, -1, 2), np.array ([1,5]))
gamma_range = gamma_range.flatten ()
```

```
Obviamente, você pode ampliar o intervalo de parâmetros, mas isso aumentará o tempo de computação.
```

![Espaço de parâmetro do SVM RBF](https://plon.io/files/58d3af091b12ce00012bd6e1)

A pesquisa em grade é um processo demorado, para que você possa usar meus melhores parâmetros
(do intervalo c = [0,1,5], gama = [0,01,0,05]):
* C = 5
* gama = 0,05
* precisão = 0,9852


```
Confusion matrix:
[[1014    0    2    0    0    2    2    0    1    3]
 [   0 1177    2    1    1    0    1    0    2    1]
 [   2    2 1037    2    0    0    0    2    5    1]
 [   0    0    3 1035    0    5    0    6    6    2]
 [   0    0    1    0  957    0    1    2    0    3]
 [   1    1    0    4    1  947    4    0    5    1]
 [   2    0    1    0    2    0 1076    0    4    0]
 [   1    1    8    1    1    0    0 1110    2    4]
 [   0    4    2    4    1    6    0    1 1018    1]
 [   3    1    0    7    5    2    0    4    9  974]]
Accuracy=0.985238095238
```

Exemplos (papers and software):

* [Pegasos](http://ttic.uchicago.edu/~nati/Publications/PegasosMPB.pdf)
* [Stochastic gradient descent](http://leon.bottou.org/projects/sgd)
* [Averaged Stochastic gradient descent](https://arxiv.org/abs/1107.2490)
* [Liblinear](https://www.csie.ntu.edu.tw/~cjlin/liblinear/)
* [Stochastic Gradient Descent with Barzilai–Borwein update step for SVM](http://www.sciencedirect.com/science/article/pii/S0020025515002467)
* [Primal SVM by Olivier Chappelle](http://olivier.chapelle.cc/primal/) - there also exists [Primal SVM in Python](https://github.com/ksopyla/primal_svm)
