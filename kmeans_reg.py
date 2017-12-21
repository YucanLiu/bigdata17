import numpy as np
import csv
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils import check_random_state

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
        temp_train_data = corr_train_data[gap-d:len(train_data)-d].transpose()
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
    print(" ")
    accuracy = []
    accuracy.append(RidgeRegression(corr_train_data,test_data))
    accuracy.append(LassoRegression(corr_train_data,test_data))
    accuracy.append(RandomForestRegreesion(corr_train_data,test_data))
    accuracy.append(BoostingRegression(corr_train_data,test_data))
    return accuracy

def RidgeRegression(train_data, train_label):
    accuracy1 = 0
    print('Ridge Regression:')
    for i in tqdm(range(max_iter_days)):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg1 = Ridge()
        reg1 = reg1.fit(X_train, Y_train)
        predict1 = reg1.predict(X_test)
        if (predict1 <= 0 and Y_test <= 0) or (predict1 > 0 and Y_test > 0):
            accuracy1 = accuracy1 + 1
    accuracy1 = accuracy1 / max_iter_days
    print("Ridge Regression Accuracy is %f" % accuracy1)
    print(" ")
    return accuracy1

def LassoRegression(train_data, train_label):
    accuracy2 = 0
    print('Lasso Regression:')
    for i in tqdm(range(max_iter_days)):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg2 = Lasso()
        reg2 = reg2.fit(X_train, Y_train)
        predict2 = reg2.predict(X_test)
        if (predict2 <= 0 and Y_test <= 0) or (predict2 > 0 and Y_test > 0):
            accuracy2 = accuracy2 + 1
    accuracy2 = accuracy2 / max_iter_days
    print("Lasso Regression Accuracy is %f" % accuracy2)
    print(" ")

    return accuracy2

def RandomForestRegreesion(train_data, train_label):
    accuracy3 = 0
    print('Ramdom Forest Regression:')
    for i in tqdm(range(max_iter_days)):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg3 = RandomForestRegressor()
        Y_train1 = np.ravel(Y_train)
        reg3 = reg3.fit(X_train, Y_train1)
        predict3 = reg3.predict(X_test)
        if (predict3 <= 0 and Y_test <= 0) or (predict3 > 0 and Y_test > 0):
            accuracy3 = accuracy3 + 1
    accuracy3 = accuracy3 / max_iter_days
    print("Random Forest Regression Accuracy is %f" % accuracy3)
    print(" ")

    return accuracy3

def BoostingRegression(train_data, train_label):
    accuracy4 = 0
    print("XGBoost Regression:")
    for i in tqdm(range(max_iter_days)):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label[260 + i : 261 + i]

        reg4 = XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.6, colsample_bytree=0.6,
                            reg_alpha=1e-5)
        reg4 = reg4.fit(X_train, Y_train)
        predict4 = reg4.predict(X_test)
        if (predict4 <= 0 and Y_test <= 0) or (predict4 > 0 and Y_test > 0):
            accuracy4 = accuracy4 + 1
    accuracy4 = accuracy4 / max_iter_days
    print("XGBoost Regression Accuracy is %f " % accuracy4)
    print(" ")

    return accuracy4

def propose_best_reg():
    max1 = 0
    pos1 = []
    max2 = 0
    pos2 = []
    regressor = ["Ridge Regression", "Lasso Regression", "Random Forest Regression", "XGBoost Regression"]
    for i in range(len(accuracy_kmeans)):
        for j in range(len(accuracy_kmeans[i])):
            if accuracy_reg[i][j] > max1:
                max1 = accuracy_reg[i][j]
                pos1 = [i,j]

    for i in range(len(accuracy_kmeans)):
        for j in range(len(accuracy_kmeans[i])):
            if accuracy_kmeans[i][j] > max2:
                max2 = accuracy_kmeans[i][j]
                pos2 = [i,j]

    if max2 > max1:
        print("Maximum Accuracy is %4.2f%%" % (max2 * 100))
        print("The best regressor is " + regressor[pos2[1]] + \
              " with k-means with a " + str(delay[pos2[0]]) +" day delay.")
    else:
        print("Maximum Accuracy is %4.2f%%" % (max1 * 100))
        print("The best regressor is " + regressor[pos1[1]] + \
              " without k-means with a " + str(delay[pos1[0]]) + " day delay.")

if __name__=='__main__':

    ###PARAMETER
    delay = [5, 20, 40, 60, 100, 120]  # date of delays
    company_name = 'BABA' # target company name

    n_c = 10  # number of clusters
    iters = 5000  # number of iterations

    train_data_path = 'data_all/output.csv'
    test_data_path = 'data_all/' + company_name + '.csv'

    #Row = time, Columns = companies
    samples, target = import_data( )
    train_data = samples
    train_label = target

    train_data_days = len(train_data)
    test_data_days = len(train_label)
    gap = train_data_days - test_data_days
    max_iter_days = test_data_days - 260

    # print("Train Data size: " + str(len(train_data)) + " * " + str(len(train_data[0])))
    # print("Test Data size: " + str(len(train_label)) + " * " + str(len(train_label[0])))

    accuracy_reg = []
    accuracy_kmeans = []
    for d in delay:
        print('Delay Days: %d' % d)

        #Traditional Regressions
        accuracy_temp = []
        accuracy_temp.append(RidgeRegression(train_data, train_label))
        accuracy_temp.append(LassoRegression(train_data, train_label))
        accuracy_temp.append(RandomForestRegreesion(train_data, train_label))
        accuracy_temp.append(BoostingRegression(train_data, train_label))
        accuracy_reg.append(accuracy_temp)

        #Kmeans + Regression
        accuracy_kmeans.append(cluster_test(train_data, train_label))
        print(' ')

    propose_best_reg()

    filename = '/home/yuxin/Documents/Bigdata/bigdata17/result/kmeans/'+company_name
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(delay)):
            writer.writerow(['day = '+ str(delay[i])])
            writer.writerow(accuracy_reg[i])
            writer.writerow(accuracy_kmeans[i])
            writer.writerow('')

