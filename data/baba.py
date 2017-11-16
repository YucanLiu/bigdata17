import csv

baba = csv.reader(open("BABA.csv"), delimiter=",")

RESULTS = []
num = 0;
for row in baba:
    # print row
    if (num != 0):
        print float(row[4]), float(row[1]), float(row[4]) - float(row[1])
        temp = []
        temp.append(float(row[4]) - float(row[1]))
        RESULTS.append(temp)
    num = num + 1

print "Total number of data: " + str(num)
print RESULTS

with open("target.csv",'wb') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(RESULTS)
