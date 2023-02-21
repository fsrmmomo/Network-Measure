from utils.util import *

tasks = []
capacity = dict()
resource = 1


class Task:
    def __init__(self, ID, usage):
        self.ID = ID
        self.usage = usage
        self.weight = 0


def read_capaticy():
    clist = dict()
    with open("../../data/capacity", "r") as f:
        for line in f.readlines():
            sw = line.split()[0]
            c = []
            for i in range(resource):
                c.append(int(line.split()[i+1]))
            clist[sw] = c
    return clist

def algo():
    global capacity
    capacity = read_capaticy()

    task1 = Task(0,[1])
    tasks.append(task1)


    build_graph(capacity)

def build_graph(capacity):
    task_num = len(tasks)
    print(tasks)
    routing = get_routing()
    sw_node = get_sw_node_list()
    node_num = 1 + 1 + len(sw_node)*2 + task_num * len(routing)

    print(node_num)
    adj = [[[0 for k in range(resource)] for i in range(node_num)] for j in range(node_num)]

    # adj = []

    index = 0
    # 添加起点  0
    node_map = []
    node_find = dict()
    node_map.append("start")
    node_find["start"] = index
    index += 1

    # 添加终点  1
    node_map.append("end")
    node_find["end"] = index
    index += 1

    # 添加 sw 点
    for sw in sw_node:
        node_map.append(sw)
        node_find[sw] = index
        index += 1

        name = sw + "x"
        node_map.append(name)
        node_find[name] = index

        adj[index - 1][index] = capacity[sw]
        adj[index][1] = capacity[sw]
        index += 1

    # 按照优先级排序
    sort_task_and_subflow()

    # node_list.append(index)
    # 添加流点
    for taskID in range(task_num):
        for i, r in enumerate(routing):
            name = "task " + str(taskID) + " subflow " + str(i)
            node_map.append(name)
            node_find[name] = index
            # 添加start到subflow的边
            # adj[0][index] = r[-1]
            adj[0][index] = [r[-1] * j for j in tasks[taskID].usage]

            # 添加subflow到对应sw的边
            for sw in r[:-1]:
                adj[index][node_find[sw]] = [r[-1] * j for j in tasks[taskID].usage]
            index += 1

    print(len(adj))
    for n,nn in enumerate(adj):
        print(str(n)+": ",end="")
        print(nn)


def max_flow():
    pass

# 实际上我们只需要排序task 因为同一个task里面的任务肯定能够最大化，不需要排序
def sort_task_and_subflow():
    if len(tasks) == 1:
        return
    total_c = [sum([capacity[j][i] for j in range(len(capacity))]) for i in range(resource)]
    weight = [0 for i in range(len(tasks))]
    for j in range(len(tasks)):
        for i in range(resource):
            weight[j] += tasks[j][i] / total_c[i]
        tasks[j].weight = weight[j]
    tasks.sort(key=lambda x: x.weight)

if __name__ == '__main__':
    algo()