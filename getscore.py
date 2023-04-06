file = './result/result_aiops.txt'
f = open(file)
sum_f1 = 0.0
cnt = 0
for line in f:
    lis = line.strip().split(' ')
    sum_f1 += float(lis[6])
    cnt+=1
print(cnt)
print(sum_f1/cnt)
f.close()