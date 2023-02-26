import pickle as pkl
from utils.params import *

def pkl_write(filename, obj):
    with open(filename, 'wb') as f:
        pkl.dump(obj, f)


def pkl_read(filename):
    obj = None
    with open(filename, 'rb') as f:
        obj = pkl.load(f)
    return obj

def get_node_list():
    node_list = []
    for i in range(1, 9):
        node_list.append("access" + str(i))
        node_list.append("merge" + str(i))
    for i in range(1, 5):
        node_list.append("core" + str(i))
    return node_list

def get_sw_node_list():
    node_list = []
    for i in range(1, access_node+1):
        node_list.append("access" + str(i))
    for i in range(1, merge_node+1):
        node_list.append("merge" + str(i))
    for i in range(1, 5):
        node_list.append("core" + str(i))
    return node_list

def get_routing():
    routing = []
    with open("../../data/ecmp_result", 'r') as f:
        for line in f.readlines():
            tmp = line.split()
            if len(tmp) == 5:
                tmp = tmp[:3]
                tmp.append(0.1)
                # tmp.append([0.2])
                routing.append(tmp[:])
            else:
                tmp = tmp[:5]
                tmp.append(0.1)
                routing.append(tmp[:])
    return routing

def gen_capacity():
    c = [1]
    with open("../../data/capacity", "w") as f:
        sws =  get_sw_node_list()

        for sw in sws:
            txt = sw
            for cc in c:
                txt += " " + str(cc)
            txt += "\n"
            f.write(txt)


if __name__ == '__main__':
    gen_capacity()