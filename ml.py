import numpy as np
import csv
from sklearn.cluster import KMeans
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

train_data_path = 'test1/samples.csv'
test_data_path = 'test1/target.csv'

#import data
def import_data( ):
    train_data = []
    test_data = []
    with open(train_data_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            row = [float(x) for x in row]
            train_data.append(row)
    train_data = np.asarray(train_data).transpose()
    # print(train_data)

    with open(test_data_path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', strict=True)
        for row in spamreader:
            row = [float(x) for x in row]
            test_data.append(row)
    test_data = np.asarray(test_data)
    return train_data, test_data


#Testing data row 126 364 409 bad data
def cluster_test(train_data,test_data):

    clf = KMeans(n_clusters=n_c, max_iter=iters, algorithm='full')
    clf = clf.fit(train_data)
    train_label = clf.predict(train_data)
    # print(train_label)
    test_label = clf.predict(test_data)
    # print(test_label)

    corr_test_data = []
    for i in range(len(train_data)):
        if train_label[i] == test_label[0]:
            corr_test_data.append(train_data[i])
    print("total number of relatives: " + str(len(corr_test_data)))

    corr_test_data = np.asarray(corr_test_data)

    LassoRegression(corr_test_data.transpose(),test_data.transpose())

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
    accuracy1 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg1 = Ridge()
        reg1 = reg1.fit(X_train, Y_train)
        predict1 = reg1.predict(X_test)
        if (predict1 <= 0 and Y_test <= 0) or (predict1 > 0 and Y_test > 0):
            accuracy1 = accuracy1 + 1
    accuracy1 = accuracy1 / (i+1)
    print("Ridge Regressor Accuracy is %f" % accuracy1)

def LassoRegression(train_data, train_label):
    accuracy2 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg2 = Lasso()
        reg2 = reg2.fit(X_train, Y_train)
        predict2 = reg2.predict(X_test)
        if (predict2 <= 0 and Y_test <= 0) or (predict2 > 0 and Y_test > 0):
            accuracy2 = accuracy2 + 1
    accuracy2 = accuracy2 / (i+1)
    print("Lasso Regressor Accuracy is %f" % accuracy2)

def RandomForestRegreesion(train_data, train_label):
    accuracy3 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg3 = RandomForestRegressor()
        Y_train1 = np.ravel(Y_train)
        reg3 = reg3.fit(X_train, Y_train1)
        predict3 = reg3.predict(X_test)
        if (predict3 <= 0 and Y_test <= 0) or (predict3 > 0 and Y_test > 0):
            accuracy3 = accuracy3 + 1
    accuracy3 = accuracy3 / (i+1)
    print("Random Forest Regressor Accuracy is %f" % accuracy3)

def BoostingRegression(train_data, train_label):
    accuracy4 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg4 = XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.6, colsample_bytree=0.6,
                            reg_alpha=1e-5)
        reg4 = reg4.fit(X_train, Y_train)
        predict4 = reg4.predict(X_test)
        if (predict4 <= 0 and Y_test <= 0) or (predict4 > 0 and Y_test > 0):
            accuracy4 = accuracy4 + 1
    accuracy4 = accuracy4 / (i + 1)
    print("XGBoost Regressor Accuracy is %f " % accuracy4)

if __name__=='__main__':
    ###PARAMETER
    d = 5  # date of delays
    n_c = 200  # number of clusters
    iters = 5000  # number of iterations

    #Row = time, Columns = companies
    SP500, target = import_data( )
    train_data = SP500
    train_label = target

    #Traditional Regressions
    # RidgeRegression()
    # LassoRegression()
    # RandomForestRegreesion()
    # BoostingRegression()

    #Kmeans + Regression
    cluster_test(train_data.transpose(), train_label.transpose())
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

# Tecniques
# Bussiness
# Big Data Aspect
