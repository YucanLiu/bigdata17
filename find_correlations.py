import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

def RidgeRegression(train_data, train_label):
    accuracy1 = 0
    for i in range(max_iter_days):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label.get_value(col=train_label.keys()[0], index= 260 + i)
        # Y_test = train_label[train_label.keys()][260 + i : 261 + i]
        Y_test_pre = train_label.get_value(col=train_label.keys()[0], index= 260 + i - 1)

        reg1 = Ridge()
        reg1 = reg1.fit(X_train, Y_train)
        predict1 = reg1.predict(X_test)
        if (predict1 - Y_test_pre <= 0 and Y_test - Y_test_pre <= 0) or (predict1 - Y_test_pre > 0 and Y_test - Y_test_pre > 0):
            accuracy1 = accuracy1 + 1
    accuracy1 = accuracy1 / max_iter_days
    print("Ridge Regressor Accuracy is %f" % accuracy1)
    return accuracy1

def LassoRegression(train_data, train_label):
    accuracy2 = 0
    for i in range(max_iter_days):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label.get_value(col=train_label.keys()[0], index= 260 + i)
        Y_test_pre = train_label.get_value(col=train_label.keys()[0], index= 260 + i - 1)

        reg2 = Lasso()
        reg2 = reg2.fit(X_train, Y_train)
        predict2 = reg2.predict(X_test)
        if (predict2 - Y_test_pre <= 0 and Y_test - Y_test_pre <= 0) or (predict2 - Y_test_pre > 0 and Y_test - Y_test_pre > 0):
            accuracy2 = accuracy2 + 1
    accuracy2 = accuracy2 / max_iter_days
    print("Lasso Regressor Accuracy is %f" % accuracy2)
    return accuracy2

def RandomForestRegreesion(train_data, train_label):
    accuracy3 = 0
    for i in range(max_iter_days):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label.get_value(col=train_label.keys()[0], index= 260 + i)
        Y_test_pre = train_label.get_value(col=train_label.keys()[0], index= 260 + i - 1)

        reg3 = RandomForestRegressor()
        Y_train1 = np.ravel(Y_train)
        reg3 = reg3.fit(X_train, Y_train1)
        predict3 = reg3.predict(X_test)
        if (predict3 - Y_test_pre <= 0 and Y_test - Y_test_pre <= 0) or (predict3 - Y_test_pre > 0 and Y_test - Y_test_pre > 0):
            accuracy3 = accuracy3 + 1
    accuracy3 = accuracy3 / max_iter_days
    print("Random Forest Regressor Accuracy is %f" % accuracy3)
    return accuracy3

def BoostingRegression(train_data, train_label):
    accuracy4 = 0
    for i in range(max_iter_days):
        X_train = train_data[gap + i - d : gap + 260 + i - d]
        Y_train = train_label[i : 260 + i]
        X_test = train_data[gap + 260 + i - d : gap + 261 + i - d]
        Y_test = train_label.get_value(col=train_label.keys()[0], index= 260 + i)
        Y_test_pre = train_label.get_value(col=train_label.keys()[0], index= 260 + i - 1)

        reg4 = XGBRegressor(max_depth=4, min_child_weight=4, gamma=0.4, subsample=0.6, colsample_bytree=0.6,
                            reg_alpha=1e-5)
        reg4 = reg4.fit(X_train, Y_train)
        predict4 = reg4.predict(X_test)
        if (predict4 - Y_test_pre <= 0 and Y_test - Y_test_pre <= 0) or (predict4 - Y_test_pre > 0 and Y_test - Y_test_pre > 0):
            accuracy4 = accuracy4 + 1
    accuracy4 = accuracy4 / max_iter_days
    print("XGBoost Regressor Accuracy is %f " % accuracy4)
    return accuracy4

def cointer(stock, target):
    sample_stock = stock[179 - d: 179 + 813 - d + 1]
    n = sample_stock.shape[1]
    keys = sample_stock.keys()
    pvalue_max = 0.1
    num_corr = 100
    while num_corr > 20:
        corr_comp = []
        for i in range(n):
            stock1 = target
            stock2 = sample_stock[keys[i]]
            result = sm.tsa.stattools.coint(stock1, stock2)
            pvalue = result[1]
            if pvalue < pvalue_max:
                corr_comp.append(keys[i])
        num_corr = len(corr_comp)
        pvalue_max = pvalue_max - 0.01
    corr_comp_data = stock[corr_comp]
    accuracy = []
    accuracy.append(RidgeRegression(corr_comp_data, target))
    accuracy.append(LassoRegression(corr_comp_data, target))
    accuracy.append(RandomForestRegreesion(corr_comp_data, target))
    accuracy.append(BoostingRegression(corr_comp_data, target))
    return accuracy

if __name__=='__main__':
    ###PARAMETER
    delay = [5, 20, 40, 60, 100, 120]  # date of delays
    # delay = [5]
    n_c = 10  # number of clusters
    iters = 5000  # number of iterations

    train_data = pd.read_csv('/home/yuxin/Documents/Bigdata/bigdata17/close_data/output_close.csv', index_col=0).dropna(axis=1)
    train_label = pd.read_csv('/home/yuxin/Documents/Bigdata/bigdata17/close_data/BABA_close.csv')

    #Row = time, Columns = companies

    train_data_days, comp1 = train_data.shape
    test_data_days, comp2 = train_label.shape
    gap = train_data_days - test_data_days
    max_iter_days = test_data_days - 260
    # print("Train Data size: " + str(train_data_days) + " * " + str(comp1))
    # print("Test Data size: " + str(test_data_days) + " * " + str(comp2))

    import csv

    for d in delay:
        print('Delay Days: %d' % d)
        accuracy = []

        #Traditional Regressions
        accuracy.append(RidgeRegression(train_data, train_label))
        accuracy.append(LassoRegression(train_data, train_label))
        accuracy.append(RandomForestRegreesion(train_data, train_label))
        accuracy.append(BoostingRegression(train_data, train_label))

        #Cointergration + Regression
        print('Cointergration + Regression: ')
        accuracy.extend(cointer(train_data, train_label))
        print(accuracy)
        print(' ')
        filename = str(d) + '.csv'
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(accuracy)

# d = 30
# best = stock['WF.csv'][179-d : (179+813-d+1-400)]
# target = target[:len(target)-400]
# x1 = range(len(best))
# x2 = range(d,len(best)+d)
# print(x2)
# import matplotlib.pyplot as plt
# plt.title('Delay = 30 days')
# line1 = plt.plot(x1, best*2, 'b', label = '1')
# line2 = plt.plot(x2, target/3 , 'r')
# plt.legend(['Wells Fargo','Alibaba'])
# plt.axis([0, 445, 0, 95])
# plt.grid(True)
# plt.show()

# plt.figure(1)
# plt.title('Histogram of IQ')
# plt.subplot(211)
# plt.plot(x1, best, 'b')
# plt.grid(True)
# plt.xlabel('Stock Price')
# plt.ylabel('Days')
# plt.legend('WellsFargo')
#
# plt.subplot(212)
# plt.plot(x2, target, 'r', label='Line 2')
# plt.grid(True)
# plt.xlabel('Stock Price')
# plt.ylabel('Days')
# plt.show()

#d = 40, MLP.csv
#max correlation day = 50
