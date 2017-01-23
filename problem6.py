#! /usr/bin/env python

from sklearn import linear_model

clf=linear_model.LogisticRegression(C=1e5, verbose=True)

nsamples, nx, ny = train_dataset.shape
nlables,= train_labels.shape
d2_train_dataset = train_dataset.reshape((nsamples,nx*ny))

clf.fit(d2_train_dataset, train_labels)

ntests, nx, ny = test_dataset.shape
ntests,= test_labels.shape
d2_test_dataset = test_dataset.reshape((ntests,nx*ny))


clf.score(d2_train_dataset, train_labels)
clf.score(d2_test_dataset, test_labels)


clf.predict(d2_test_dataset[1])
print(test_labels[1])


clf_pickle_file = 'clf.pickle'

try:
  f = open(clf_pickle_file, 'wb')
  pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
  f.close()
except Exception as e:
  print('Unable to save data to', clf_pickle_file, ':', e)
  raise



try:
  f = open(clf_pickle_file, 'rb')
  clf2= pickle.load(f)
  f.close()
except Exception as e:
  print('Unable to load data ', pickle_file, ':', e)
  raise

