'''MNIST classification using Support Vector algorithm with RBF kernel
 all parameters are optimized  by random search with cross-validation '''
# Author: Krzysztof Sopyla <krzysztofsopyla@gmail.com>
# https://ksopyla.com
# License: MIT

# Standard scientific Python imports
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime as dt


from time import time



from sklearn.model_selection import RandomizedSearchCV



# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# import custom module
from mnist_helpers import *


#fetch original mnist dataset
from sklearn.datasets import fetch_mldata
# it creates mldata folder in your root project folder
mnist = fetch_mldata('MNIST original', data_home='./')


#minist object contains: data, COL_NAMES, DESCR, target fields
#you can check it by running
mnist.keys()

#data field is 70k x 784 array, each row represents pixels from 28x28=784 image
images = mnist.data
targets = mnist.target

# Let's have a look at the random 16 images, 
# We have to reshape each data row, from flat array of 784 int to 28x28 2D array

#pick  random indexes from 0 to size of our dataset
show_some_digits(images,targets)

#---------------- classification begins -----------------
#scale data for [0,255] -> [0,1]
#sample smaller size for testing
#rand_idx = np.random.choice(images.shape[0],10000)
#X_data =images[rand_idx]/255.0
#Y      = targets[rand_idx]

#full dataset classification
X_data = images/255.0
Y = targets

#split data to train and test
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, Y, test_size=0.15, random_state=42)


############### Classification with grid search ##############

# Warning! It takes really long time to compute this about 2 days

# Create parameters grid for RBF kernel, we have to set C and gamma

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

C_range = sp_uniform(scale=10)
gamma_range = sp_uniform(scale=1)
parameters = {'kernel':['rbf'],
              'C':C_range, 
              'gamma': gamma_range
 }

n_iter_search = 10
svm_clsf = svm.SVC()
rnd_clsf = RandomizedSearchCV(estimator=svm,
                              param_distributions=parameters,
                              n_iter=n_iter_search, cv=5,
                              verbose=2)

start_time = dt.datetime.now()
print('Start param searching at {}'.format(str(start_time)))

rnd_clsf.fit(X_train, y_train)

elapsed_time= dt.datetime.now() - start_time
print('Elapsed time, param searching {}'.format(str(elapsed_time)))
sorted(rnd_clsf.cv_results_.keys())

classifier = rnd_clsf.best_estimator_
params = rnd_clsf.best_params_



scores = rnd_clsf.cv_results_['mean_test_score'].reshape(len(C_range),
                                                     len(gamma_range))

plot_param_space_scores(scores, C_range, gamma_range)


######################### end grid section #############

# Now predict the value of the test
expected = y_test
predicted = classifier.predict(X_test)

show_some_digits(X_test,predicted,title_text="Predicted {}")

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
      
cm = metrics.confusion_matrix(expected, predicted)
print("Confusion matrix:\n%s" % cm)

plot_confusion_matrix(cm)

print("Accuracy={}".format(metrics.accuracy_score(expected, predicted)))


