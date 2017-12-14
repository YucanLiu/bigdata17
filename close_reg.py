import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

train_data_path = 'close_data/output_close.csv'
# train_data_path = 'test1/samples.csv'
test_data_path = 'close_data/BABA_close.csv'

class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means

    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """

    def __init__(self, n_clusters=3, max_iter=50, tol=1e-3, random_state=None,
                 kernel="linear", gamma=None, degree=3, coef0=1,
                 kernel_params=None, verbose=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.verbose = verbose

    @property
    def _pairwise(self):
        return self.kernel == "precomputed"

    def _get_kernel(self, X, Y=None):
        if callable(self.kernel):
            params = self.kernel_params or {}
        else:
            params = {"gamma": self.gamma,
                      "degree": self.degree,
                      "coef0": self.coef0}
        return pairwise_kernels(X, Y, metric=self.kernel,
                                filter_params=True, **params)

    def fit(self, X, y=None, sample_weight=None):
        n_samples = X.shape[0]

        K = self._get_kernel(X)

        sw = sample_weight if sample_weight else np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(self.random_state)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                if self.verbose:
                    print
                    "Converged at iteration", it + 1
                break

        self.X_fit_ = X

        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            # if np.sum(mask) == 0:
            #     raise ValueError("Empty cluster found, try smaller n_cluster.")

            denom = sw[mask].sum()
            denomsq = denom * denom

            if update_within:
                KK = K[mask][:, mask]  # K[mask, mask] does not work.
                dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                within_distances[j] = dist_j
                dist[:, j] += dist_j
            else:
                dist[:, j] += within_distances[j]

            dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom

    def predict(self, X):
        K = self._get_kernel(X, self.X_fit_)
        n_samples = X.shape[0]
        dist = np.zeros((n_samples, self.n_clusters))
        self._compute_dist(K, dist, self.within_distances_,
                           update_within=False)
        return dist.argmin(axis=1)

#import data
def import_data( ):
    # Rows: Days; Columns: Companies => 993 * 998
    train_data = []
    test_data = []
    with open(train_data_path) as csvfile:
        datareader = csv.reader(csvfile, delimiter=',', strict = True)
        i = 0
        for row in datareader:
            if(i != 0):
                row = [float(x) for x in row if x != '']
                train_data.append(row)
            i += 1
    train_data = np.asarray(train_data) #993 days * 998 companies

    with open(test_data_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', strict=True)
        for row in spamreader:
            row = [float(x) for x in row]
            test_data.append(row)
    test_data = np.asarray(test_data) # 814 days
    # print(len(test_data))
    #testing data is a matrix (days * companies)
    #training data is a cloumn
    return train_data, test_data


def cluster_test(train_data,test_data):
    # clf = KMeans(n_clusters=n_c, max_iter=iters, algorithm='full')
    # print("Train Data size: " + str(len(train_data)) + " * " + str(len(train_data[0])))
    # print("Test Data size: " + str(len(test_data)) + " * " + str(len(test_data[0])))

    corr_train_data = train_data
    corr_data_num = 100

    while corr_data_num > 20:
        temp_train_data = corr_train_data[183-d:997-d].transpose()
        # print("Temp Train Data size: " + str(len(temp_train_data)) + " * " + str(len(temp_train_data[0])))

        clf = KernelKMeans(n_clusters=n_c, max_iter=5000)
        clf = clf.fit(temp_train_data)
        train_label = clf.predict(temp_train_data)
        # print("Train Label size: " + str(len(train_label)))
        test_label = clf.predict(test_data.transpose())
        # print("Test Label size: " + str(len(test_label)))

        corr_train_data = corr_train_data.transpose()
        corr_train_data_tmp = []
        for i in range(len(corr_train_data)):
            if train_label[i] == test_label[0]:
                corr_train_data_tmp.append(corr_train_data[i])
        # print("total number of relatives: " + str(len(corr_train_data_tmp)))
        corr_data_num = len(corr_train_data_tmp)
        # temp_train_data = np.asarray(corr_test_data).transpose()

        corr_train_data = np.asarray(corr_train_data_tmp).transpose()
        # print("Correlating Test Data size: " + str(len(corr_train_data)) + " * " + str(len(corr_train_data[0])))

    print("KMeans + Regression Result: ")
    RidgeRegression(corr_train_data,test_data)
    LassoRegression(corr_train_data,test_data)
    RandomForestRegreesion(corr_train_data,test_data)
    BoostingRegression(corr_train_data,test_data)

    # plt_data=np.concatenate((corr_test_data,test_data))

    # for x in range(len(plt_data)):
    #     for y in range(len(plt_data[0])):
    #         if plt_data[x][y] >= 0:
    #             plt_data[x][y] = 1
    #         else:
    #             plt_data[x][y] = -1
    # print(plt_data)

    #sum=0
    #for x in range(len(plt_data)):
    #    accuracy = np.mean(plt_data[x] == plt_data[len(plt_data)-1])
    #    if(accuracy != 1.0):
    #        sum+=accuracy
    #    # print(accuracy)
    # print(sum)
    # print('average accuracy:',sum/(len(plt_data)-1))
    # return sum/(len(plt_data)-1)

    #import matplotlib.pyplot as mpl
    # x = np.asarray([range(len(corr_test_data))])
    #plt_data=plt_data.transpose()
    #mpl.plot(plt_data,".")
    #mpl.grid(True)

    #mpl.show()

def RidgeRegression(train_data, train_label):
    mae1 = 0
    for i in range(r):
        X_train = train_data[183 + i - d : 183 + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[183 + 260 + i - d : 183 + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg1 = Ridge()
        reg1 = reg1.fit(X_train, Y_train)
        predict1 = reg1.predict(X_test)
        mae1 =  mae1 + abs(predict1 - Y_test)
    mae1 = mae1/r
    print("Ridge Regressor MAE is %f" % mae1)

def LassoRegression(train_data, train_label):
    mae2 = 0
    for i in range(r):
        X_train = train_data[183 + i - d : 183 + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[183 + 260 + i - d : 183 + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg2 = Lasso(max_iter=2000)
        reg2 = reg2.fit(X_train, Y_train)
        predict2 = reg2.predict(X_test)
        mae2 =  mae2 + abs(predict2 - Y_test)
    mae2 = mae2/r
    print("Lasso Regressor MAE is %f" % mae2)

def RandomForestRegreesion(train_data, train_label):
    mae3 = 0
    for i in range(r):
        X_train = train_data[183 + i - d : 183 + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[183 + 260 + i - d : 183 + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg3 = RandomForestRegressor()
        Y_train1 = np.ravel(Y_train)
        reg3 = reg3.fit(X_train, Y_train1)
        predict3 = reg3.predict(X_test)
        mae3 = mae3 + abs(predict3 - Y_test)
    mae3 = mae3 / r
    print("Random Forest Regressor MAE is %f" % mae3)

def BoostingRegression(train_data, train_label):
    mae4 = 0
    for i in range(r):
        X_train = train_data[183 + i - d : 183 + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[183 + 260 + i - d : 183 + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg4 = XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.6, colsample_bytree=0.6,
                            reg_alpha=1e-5)
        reg4 = reg4.fit(X_train, Y_train)
        predict4 = reg4.predict(X_test)
        mae4 = mae4 + abs(predict4 - Y_test)
    mae4 = mae4 / r
    print("XGBoost Regressor MAE is %f" % mae4)

if __name__=='__main__':
    ###PARAMETER
    # delay = [5, 20, 40, 60, 100, 120]  # date of delays
    delay = [5]
    n_c = 10  # number of clusters
    iters = 5000  # number of iterations
    r = 10

    #Row = time, Columns = companies
    samples, target = import_data( )
    train_data = samples
    train_label = target
    print("Train Data size: " + str(len(train_data)) + " * " + str(len(train_data[0])))
    print("Test Data size: " + str(len(train_label)) + " * " + str(len(train_label[0])))

    for d in delay:
        print('Delay Days: %d' % d)
        #Traditional Regressions
        RidgeRegression(train_data, train_label)
        LassoRegression(train_data, train_label)
        RandomForestRegreesion(train_data, train_label)
        BoostingRegression(train_data, train_label)

        #Kmeans + Regression
        cluster_test(train_data, train_label)
        print(' ')

    #max = 0
    #max_delay = 0
    #for x in range(1,100):
    #    d = 96
    #    print('\ncurrent date of delays:', x)
    #    accuracy = cluster_test()
    #    if(accuracy > max):
    #        max = accuracy
    #        max_delay = x

    #print ('\nThe best accuracy is:', max, 'when delay is:', max_delay)

# Techniques
# Bussiness
# Big Data Aspect
