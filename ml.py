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
def cluster_test(train_data, test_data):
    t_data = []
    for x in range (0, len(train_data)):
        t_data.append(np.asarray(train_data[x][: len(train_data[0]) - d]))
    train_data = np.asarray(t_data)
    # print(len(train_data[0]))

    # print("original test data")
    # print(test_data.shape)

    test_data = test_data[0][d:]
    temp_data = []
    temp_data.append(test_data)
    test_data = np.asarray(temp_data)
    # print(len(test_data))
    # print("final test data")
    # print(test_data.shape)


    ### DEBUGGING
    # print(train_data.shape)
    # print(test_data.shape)

    ######K-means
    #data = np.concatenate((test_data,train_data))
    #print(data)
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

    corr_test_data=np.asarray(corr_test_data)
    plt_data=np.concatenate((corr_test_data,test_data))

    for x in range(len(plt_data)):
        for y in range(len(plt_data[0])):
            if plt_data[x][y] >= 0:
                plt_data[x][y] = 1
            else:
                plt_data[x][y] = -1
    # print(plt_data)

    sum=0
    for x in range(len(plt_data)):
        accuracy = np.mean(plt_data[x] == plt_data[len(plt_data)-1])
        if(accuracy != 1.0):
            sum+=accuracy
        # print(accuracy)
    # print(sum)
    print('average accuracy:',sum/(len(plt_data)-1))
    return sum/(len(plt_data)-1)

    #import matplotlib.pyplot as mpl
    # x = np.asarray([range(len(corr_test_data))])
    #plt_data=plt_data.transpose()
    #mpl.plot(plt_data,".")
    #mpl.grid(True)

    #mpl.show()

def RidgeRegression():
    mae1 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg1 = Ridge()
        reg1 = reg1.fit(X_train, Y_train)
        predict1 = reg1.predict(X_test)
        mae1 = mae1 + np.absolute(Y_test - predict1)
    mae1 = mae1 / (i+1)
    print("Ridge Regressor Mean Absolute Error is %f" % mae1 )

def LassoRegression():
    mae2 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg2 = Lasso()
        reg2 = reg2.fit(X_train, Y_train)
        predict2 = reg2.predict(X_test)
        mae2 = mae2 + np.absolute(Y_test - predict2)
    mae2 = mae2 / (i+1)
    print("Lasso Regressor Mean absolute error is %f" % mae2)

def RandomForestRegreesion():
    mae3 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg3 = RandomForestRegressor()
        Y_train1 = np.ravel(Y_train)
        reg3 = reg3.fit(X_train, Y_train1)
        predict3 = reg3.predict(X_test)
        mae3 = mae3 + np.absolute(Y_test - predict3)
    mae3 = mae3 / (i+1)
    print("Random Forest Regressor Mean absolute error is %f" % mae3)

def BoostingRegression():
    mae4 = 0
    for i in range(len(train_data) - d - 240):
        X_train = train_data[i : 240+i]
        Y_train = train_label[i+d : 240+i+d]
        X_test = train_data[240+i : 240+i+1]
        Y_test = train_label[240+d : 240+d+1]

        reg4 = XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.6, colsample_bytree=0.6,
                            reg_alpha=1e-5)
        reg4 = reg4.fit(X_train, Y_train)
        predict4 = reg4.predict(X_test)
        mae4 = mae4 + np.absolute(Y_test - predict4)
    mae4 = mae4 / (i+1)
    print("XGBoost Regressor Mean absolute error is %f " % mae4)

if __name__=='__main__':
    ###PARAMETER
    d = 5  # date of delays
    n_c = 400  # number of clusters
    iters = 5000  # number of iterations

    #Row = time, Columns = companies
    SP500, target = import_data( )
    train_data = SP500
    train_label = target

    RidgeRegression()
    LassoRegression()
    RandomForestRegreesion()
    BoostingRegression()

    #Kmeans + Regression

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

# kernel k means
# Time series analysis
# R
# Random Forest, Gradient Boosting
# Ridge Regression, Lasso Regression
# Compairsion

# Tecniques
# Bussiness
# Big Data Aspect
