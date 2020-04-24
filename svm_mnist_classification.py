# MNIST classification using Support Vector algorithm with RBF kernel
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT

import datetime as dt
import time

import matplotlib.pyplot as plt
import numpy as np
# Importa datasets, classificadores e métricas de performance
from sklearn import datasets, metrics, svm
# Busca o data
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from mnist_helpers import *

mnist = fetch_openml('mnist_784')

# O objeto do minist contem: data, COL_NAMES, DESCR, target fields
# aqui imprimos rodando
mnist.keys()

# o campo 'data' é um vetor 70k x 784 array, cada linha representa em pixels 28x28=784, formando uma imagem
images = mnist.data
targets = mnist.target

# Imprimindo as imagens
# Iremos transformar num vetor (assim como na rede)
show_some_digits(images,targets)


#---------------- Inicio da classificação -----------------
# Re-escala os dados para que [0,255] -> [0,1]
# Separa em conjuntos de treinamento e teste

X_data = images/255.0
Y = targets

X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


################ Classificador ###########
# Cria o classificador SVM com os hyperparametros C e gamma

param_C = 5
param_gamma = 0.05
classifier = svm.SVC(C=param_C,gamma=param_gamma)

#We learn the digits on train part
start_time = dt.datetime.now()
print('Start learning at {}'.format(str(start_time)))
classifier.fit(X_train, y_train)
end_time = dt.datetime.now() 
print('Stop learning {}'.format(str(end_time)))
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))


########################################################
# Realiza as predições
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))
