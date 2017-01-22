#! /usr/bin/env python

hashList={}
def createHashValueList(dateset):
    for index in range(0,dateset.shape[0]):
        if index%1000==0:
            print(index)
        sample_image = dateset[index, :, :]  # extract a 2D slice
        hashValue=hash(str(sample_image))
        if hashValue in hashList:
            pass
            #print("found duplicated image in index:",index,", it duplicates with:",hashList[hashValue])
        else:
            hashList[hashValue]=index
"""
    sample_idx = np.random.randint(len(lables))
    sample_image = dateset[sample_idx, :, :]  # extract a 2D slice
    plt.figure()
    plt.imshow(sample_image)  # display it
    print(chars[lables[sample_idx]])
"""

def checkHashValue(dateset):
    for index in range(0,dateset.shape[0]):
        if index%1000==0:
            print(index)
        sample_image = dateset[index, :, :]  # extract a 2D slice
        hashValue=hash(str(sample_image))
        if hashValue in hashList:
            print("found duplicated image in index:",index,", it duplicates with:",hashList[hashValue])

print("####################################################")
print("Start to create hash list with  the test data!!!!!!!!!!!!!!!!!!")
print("####################################################")
createHashValueList(test_dataset)

checkHashValue(train_dataset)




