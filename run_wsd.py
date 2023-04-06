import os
file_list = os.listdir('./data/AIOPS2018')
file_list = sorted(file_list)
print(file_list)

for i in file_list:
    os.system('python cpu_train.py {}'.format(i))