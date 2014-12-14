import numpy as np


def sample_kmeans():
    priors = np.random.normal(loc=[0, 0], size=(5, 2), scale=0.8)
    print priors
    f=open("sample_kmeans.csv","w")
    for prior in priors:
        a = np.random.normal(loc=prior, size=(50, 2), scale=0.3)
        for data in a:
            f.write("{},{},{}\n".format(0, data[0], data[1]))
    f.close()

    
def sample_svm():
    pos_mean = np.array([-0.5, -0.5])
    neg_mean = np.array([0.5, 0.5])
    pos = np.random.normal(loc=pos_mean, size=(300,2), scale=0.5)
    neg = np.random.normal(loc=neg_mean, size=(300,2), scale=0.5)
    f=open("sample_svm.csv","w")
    for ind, samples in enumerate([neg, pos]):
        for data in  samples:
            f.write("{},{},{}\n".format(ind, data[0], data[1]))
    f.close()


sample_svm()
