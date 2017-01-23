#! /usr/bin/env python

chars=['A','B','C','D','E','F','G','H','I','J']

def displayNLetter(num, dataset, lables):
    sample_idx = num
    sample_image = dataset[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it
    print(chars[lables[sample_idx]])


def displayLetter(dataset, lables):
    sample_idx = np.random.randint(len(lables))
    displayNLetter(sample_idx, dataset, lables)
"""
    sample_image = dataset[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it
    print(chars[lables[sample_idx]])
"""
displayLetter(train_dataset,train_labels)



