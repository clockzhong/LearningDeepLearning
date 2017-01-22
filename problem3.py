#! /usr/bin/env python

def getImageLen(picke_file):
	length=0
	with open(pickle_file, 'rb') as f:
		letter_set = pickle.load(f)  # unpickle
		length=len(letter_set)
	return length

index=0
for file in train_datasets:
	print("index",index,"number: ",getImageLen(train_datasets[index]))
	index+=1
