#! /usr/bin/env python

def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data

filename ='text8.zip'
words = read_data(filename)
print('Data size %d' % len(words))
