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


if __name__ == '__main__':
    read_ms()