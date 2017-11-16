import fix_yahoo_finance as yf
import csv

sp500 = csv.reader(open("sp500"), delimiter=",")

for row in sp500:
    print row

for x in row:
    print x
    data = yf.download(x)
    print "Data Size = " + str(data.size) + "\n"
    if data.size != 0:
        data.to_csv('data/'+x+'.csv')
    else:
        print "NULL DATA"
