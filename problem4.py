#! /usr/bin/env python

chars=['A','B','C','D','E','F','G','H','I','J']

def displayLetter(dateset, lables):
    sample_idx = np.random.randint(len(lables))
    sample_image = dateset[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it
    print(chars[lables[sample_idx]])

displayLetter(train_dataset,train_labels)



