from utils.util import *
from queue import Queue

tasks = []
capacity = dict()
resource = 1
maxnum = 99999999


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
                c.append(int(line.split()[i + 1]))
            clist[sw] = c
    return clist


def algo():
    global capacity
    # 得到sw的容量
    capacity = read_capaticy()

    # 创建不同的任务
    task1 = Task(0, [1])
    tasks.append(task1)

    task_num = len(tasks)
    print(tasks)
    routing = get_routing()

    print(sum([i[-1] for i in routing]))
    sw_node = get_sw_node_list()
    node_num = 1 + 1 + len(sw_node) * 2 + task_num * len(routing)

    # 创建图 adj_copy用于对比
    adj, adj_copy = build_graph(capacity, task_num, node_num, sw_node, routing)

    ans = max_flow(adj, task_num, node_num, sw_node, routing)
    print(ans)


def build_graph(capacity, task_num, node_num, sw_node, routing):
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
    for n, nn in enumerate(adj[:10]):
        print(str(n) + ": ", end="")
        print(nn)

    adj_copy = [[adj[j][i].copy() for i in range(node_num)] for j in range(node_num)]

    for n, nn in enumerate(adj_copy[:10]):
        print(str(n) + ": ", end="")
        print(nn)
    return adj, adj_copy


def max_flow(adj, task_num, node_num, sw_node, routing):
    # 从start相邻的每个节点开始，使用BFS进行计算
    f_each_task = len(routing)
    ans = 0

    # 从优先级高的任务开始
    for i, t in enumerate(tasks):
        print(t)
        # 每个任务遍历
        index_start = 2 + len(sw_node) * 2 + i * f_each_task
        task_usage = t.usage
        for f in range(index_start, index_start + f_each_task):
            res = routing[f - index_start][-1]
            while res > 10E-7:
                pre = [0 for i in range(node_num)]
                if bfs(adj, f, 1, node_num, task_usage, pre):

                    # 回溯找到可以增广的最大路径
                    now = 1

                    # 寻找可以增广的最大流量
                    mind = float('inf')
                    while now != 0:
                        last = pre[now]
                        mind = min_capacity(adj[last][now], mind, task_usage)

                        now = last
                    mind = min(mind, res)
                    res -= mind
                    ans += mind

                    # 更新残差网络
                    now = 1
                    print(mind)
                    print(res)
                    while now != 0:
                        last = pre[now]
                        update_adj(adj, last, now, task_usage, mind, -1)
                        update_adj(adj, now, last, task_usage, mind, 1)
                        now = last
                    print("333")
                else:
                    break
    return ans


def update_adj(adj, s, t, usage, f, type):
    #  1加 -1捡
    for i in range(resource):
        if usage[i] > 0:
            adj[s][t][i] += type * usage[i] * f
            if abs(adj[s][t][i]) < 10E-7:
                adj[s][t][i] = 0


# def dfs(adj, s, t, node_num, usage, pre):
#
def bfs(adj, s, t, node_num, usage, pre):
    vis = [0 for i in range(node_num)]
    # pre = [0 for i in node_num]

    vis[0] = 1
    pre[s] = 0
    # pre[0] = 0

    queue = Queue()
    queue.put(s)
    vis[s] = 1
    while not queue.empty():
        now = queue.get()
        # 因为回到start认为总是失败的，所以直接排除0
        for i in range(1, node_num):
            if vis[i] == 0 and check_capacity(usage, adj[now][i]):
                vis[i] = 1
                pre[i] = now
                if i == t:
                    return True
                queue.put(i)
    return False


def check_capacity(usage, cap):
    for i in range(resource):
        if usage[i] > 0 and cap[i] <= 0:
            return False
    return True


def min_capacity(cap, mind, usage):
    for i in range(resource):
        if usage[i] > 0:
            mind = min(mind, cap[i] / usage[i])
    return mind


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
