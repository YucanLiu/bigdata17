import fix_yahoo_finance as yf
import csv

sp500 = csv.reader(open("sp500"), delimiter=",")

for row in sp500:
    print row

RESULTS = []
num = 0

for x in row:
    print x
    data = yf.download(x, start="2016-01-01", end="2017-01-01")
    print "Data Size = " + str(data.size) + "\n"

    if data.size != 0:
        diff=(data["Close"]-data["Open"]).values.tolist()
        RESULTS.append(diff)
    else:
        print "NULL DATA"
    # break
    num = num + 1

# print RESULTS
print "Total number of stocks: " + str(num)

with open("2016/samples.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(RESULTS)
