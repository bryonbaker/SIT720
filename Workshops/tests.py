import pandas as pd
import numpy as np

def uniqueList(list1):
    # insert the list to the set
    list_set = set(list1)
    # convert the set to the list
    unique_list = (list(list_set))
    unique_list.sort()
    return unique_list

# Finds the label with the highest frequency for a given cluster.
def findMaxLabel( data ):
    unique, counts = np.unique(data, return_counts=True)
    # print(np.asarray((unique, counts)).T)
    idx = np.argmax(counts, axis=0)
    # print(f"Index of max: {idx}")

    return unique[idx], counts[idx]

# Accepts a dataframe of cluster numbers and labels and calculates the purity of the clustering
def calculatePurity(data):
    print(data.shape)
    purity = 0

    print("Input data:\n{}".format(data))

    # Get a list of clusters that has no duplicates so we can use it as a key.
    clusterList = uniqueList(data['cluster'])
    print(f"Cluster list: {clusterList}")

    # Go through each cluster one by one.
    runningSum = 0
    for c in clusterList:
        print(f"Counting instances in cluster {c}")
        m1,m2 = findMaxLabel(data[data["cluster"] == c]["label"])
        print("Maximum instance in cluster {} is {}. It has a count of {}.".format(c,m1,m2))
        runningSum = runningSum + m2

    purity = runningSum / data.shape[0]

    return purity




#
#
#
c = pd.Series([1,1,2,1,3,2,2],name='cluster')
l = pd.Series(['class5', 'class5', 'class3', 'class1', 'class4', 'class3','class5'],name='label')
d = pd.concat([c, l], axis=1)
purity = calculatePurity(d)
print(f"Purity score is: {purity}")
