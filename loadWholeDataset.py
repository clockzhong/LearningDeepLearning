#! /usr/bin/env python

pickle_file = 'notMNIST.pickle'

try:
  f = open(pickle_file, 'rb')
  data= pickle.load(f)
  train_dataset = data['train_dataset']
  train_labels = data['train_labels']
  test_dataset = data['test_dataset']
  test_labels = data['test_labels']
  valid_dataset = data['valid_dataset']
  valid_labels = data['valid_labels']
  f.close()
except Exception as e:
  print('Unable to load data ', pickle_file, ':', e)
  raise




#statinfo = os.stat(pickle_file)
#print('Compressed pickle size:', statinfo.st_size)

