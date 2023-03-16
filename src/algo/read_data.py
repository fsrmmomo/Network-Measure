from statistics import mean

from utils.util import *
def read_ms():

    prefix = "../../result/ms-"
    data = []
    for i in range(1,6):
        filename = "../../result/ms-"+str(i)
        tmp = []
        with open(filename,'r') as f:
            for line in f.readlines():
                tmp.append(float(line[1:-2]))
                print(float(line[1:-2]))
        data.append(tmp)
    pkl_write("../../result/ms",data)

def read_ms():

    prefix = "../../result/ms-"
    data = []
    for i in range(1,6):
        filename = "../../result/ms-"+str(i)
        tmp = []
        with open(filename,'r') as f:
            for line in f.readlines():
                tmp.append(float(line[1:-2]))
                print(float(line[1:-2]))
        data.append(tmp)
    pkl_write("../../result/ms",data)

def read_mm():
    # prefix = "../../result/mm-"+str(i)+"2"
    data = []
    data.append([])
    for i in range(2,6):
        filename1 = "../../result/mm-"+str(i)+"1"
        print(filename1)
        tmp = []
        algo1 = []
        algo2 = []
        algo3 = []
        algo4 = []
        with open(filename1,'r') as f:
            for line in f.readlines():
                out = line.replace('[','').replace(']','')
                dlist = out.split(',')
                dlist = [float(d) for d in dlist]

                dots = []
                tmp1 = []
                tmp2 = []
                for j in range(len(dlist)//2):
                    tmp1.append(dlist[2*j])
                    tmp2.append(dlist[2*j+1])
                algo1.append(tmp1)
                algo2.append(tmp2)
                #     dots.append(min(1,dlist[2*i]/dlist[2*i+1]))
                # print(dots)
                # print(mean(dots))
                # tmp.append(mean(dots))
        filename2 = "../../result/mm-" + str(i) + "2"
        with open(filename2,'r') as f:
            for line in f.readlines():
                out = line.replace('[','').replace(']','')
                dlist = out.split(',')
                dlist = [float(d) for d in dlist]
                dots = []
                tmp3 = []
                tmp4 = []
                for j in range(len(dlist)//2):
                    tmp3.append(dlist[2*j])
                    tmp4.append(dlist[2*j+1])
                algo3.append(tmp3)
                algo4.append(tmp4)
        # print(algo1)
        # print(algo2)
        # print(algo3)
        # print(algo4)
        for j,d in enumerate(algo1):
            for k,v in enumerate(d):
                d[k] = v/algo4[j][k]
        for j,d in enumerate(algo2):
            for k,v in enumerate(d):
                d[k] = v/algo4[j][k]
        for j,d in enumerate(algo3):
            for k,v in enumerate(d):
                d[k] =min(1,v/algo4[j][k])
        # print(algo1)
        # print(algo2)
        # print(algo3)
        # print(algo4)
        tmp.append([mean(d) for d in algo1])
        tmp.append([mean(d) for d in algo2])
        tmp.append([mean(d) for d in algo3])
        tmp.append([1 for d in algo4])
        for t in tmp:
            print(t)
        print(tmp)
        data.append(tmp)
    print(data)
    pkl_write("../../result/mm", data)

if __name__ == '__main__':
    # read_ms()
    read_mm()