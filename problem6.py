#! /usr/bin/env python

from sklearn import linear_model

clf=linear_model.LogisticRegression(C=1e5, verbose=True)

nsamples, nx, ny = train_dataset.shape
nlables,= train_labels.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

clf.fit(d2_train_dataset, train_labels)


clf.score(d2_train_dataset, train_labels)
nsamples, nx, ny = train_dataset.shape
nlables,= train_labels.shape

clf.score(test_dataset, test_labels)

