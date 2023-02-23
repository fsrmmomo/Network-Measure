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
    sum1 = 0
    sum2 = 0
    for i in range(node_num):
        for j in range(node_num):
            sum1 += sum(adj[i][j])
            sum2 += sum(adj_copy[i][j])
    print(sum1)
    print(sum2)
    # print(adj)
    # print(adj_copy)

    ans = max_flow(adj, task_num, node_num, sw_node, routing)

    check_result(adj, adj_copy, task_num, node_num, sw_node, routing)
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

        adj[index - 1][index] = capacity[sw][:]
        adj[index][1] = capacity[sw][:]
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
    # for n, nn in enumerate(adj[:10]):
    #     print(str(n) + ": ", end="")
    #     print(nn)

    adj_copy = [[adj[j][i].copy() for i in range(node_num)] for j in range(node_num)]
    #
    # for n, nn in enumerate(adj_copy[:10]):
    #     print(str(n) + ": ", end="")
    #     print(nn)

    # adj_copy = [[adj[j][i] for i in range(node_num)] for j in range(node_num)]
    return adj, adj_copy


def max_flow(adj, task_num, node_num, sw_node, routing):
    # 从start相邻的每个节点开始，使用BFS进行计算
    f_each_task = len(routing)
    ans = 0

    # 从优先级高的任务开始 bfs
    for i, t in enumerate(tasks):
        print(t)
        # 每个任务遍历
        index_start = 2 + len(sw_node) * 2 + i * f_each_task
        task_usage = t.usage
        for f in range(index_start, index_start + f_each_task):
            res = routing[f - index_start][-1]
            while res > 1E-7:
                pre = [0 for _ in range(node_num)]
                vis = [0 for i in range(node_num)]
                # if bfs(adj, f, 1, node_num, task_usage, pre, vis):
                vis[0] = 1
                pre[f] = 0
                if dfs(adj, f, 1, f, node_num, task_usage, pre, vis):

                    # 回溯找到可以增广的最大路径

                    # 寻找可以增广的最大流量
                    mind = float('inf')
                    now = 1
                    while now != 0:
                        last = pre[now]
                        mind = min_capacity(adj[last][now], mind, task_usage)

                        now = last
                    mind = min(mind, res)
                    res -= mind
                    ans += mind

                    # 更新残差网络
                    print(mind)
                    print(res)
                    print(ans)
                    now = 1
                    while now != 0:
                        last = pre[now]
                        update_adj(adj, last, now, task_usage, mind, -1)
                        update_adj(adj, now, last, task_usage, mind, 1)
                        now = last
                    print()
                else:
                    break

    # dfs
    # for i, t in enumerate(tasks):
    #     print(t)
    #     # 每个任务遍历
    #     index_start = 2 + len(sw_node) * 2 + i * f_each_task
    #     task_usage = t.usage
    #     while True:
    #         pre = [0 for _ in range(node_num)]
    #         vis = [0 for _ in range(node_num)]
    #         # if bfs(adj, f, 1, node_num, task_usage, pre, vis):
    #         # vis[0] = 1
    #         pre[0] = 0
    #         if dfs(adj, 0, 1, 0, node_num, task_usage, pre, vis):
    #             # 回溯找到可以增广的最大路径
    #
    #             # 寻找可以增广的最大流量
    #             mind = float('inf')
    #             now = 1
    #             while now != 0:
    #                 print(now, end=" ")
    #                 last = pre[now]
    #                 mind = min_capacity(adj[last][now], mind, task_usage)
    #
    #                 now = last
    #             print()
    #             ans += mind
    #
    #             # 更新残差网络
    #             print(mind)
    #             print(ans)
    #             now = 1
    #             last = pre[now]
    #             sum1 = 0
    #             for ii in range(node_num):
    #                 for jj in range(node_num):
    #                     # sum1 += sum(adj[ii][jj])
    #                     sum1 += adj[ii][jj][0]
    #             print(sum1)
    #             while now != 0:
    #                 last = pre[now]
    #                 print(last,end=" ")
    #                 print(now)
    #                 print(adj[last][now][0])
    #                 print(adj[2][3][0])
    #                 update_adj(adj, last, now, task_usage, mind, -1.0)
    #                 update_adj(adj, now, last, task_usage, mind, 1.0)
    #                 print(adj[last][now][0])
    #                 print(adj[2][3][0])
    #                 now = last
    #
    #             sum1 = 0
    #             for ii in range(node_num):
    #                 for jj in range(node_num):
    #                     # sum1 += sum(adj[ii][jj])
    #                     sum1 += adj[ii][jj][0]
    #             print(sum1)
    #             print()
    #         else:
    #             print("no dfs")
    #             break

    return ans


def check_result(adj, adj_copy, task_num, node_num, sw_node, routing):
    print(node_num)

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

        index += 1
    sort_task_and_subflow()
    for taskID in range(task_num):
        for i, r in enumerate(routing):
            name = "task " + str(taskID) + " subflow " + str(i)
            node_map.append(name)
            node_find[name] = index
            index += 1

    # 首先检查分流量限制是否满足

    index_shift = 42
    for taskID in range(task_num):
        for i, r in enumerate(routing):
            for res in range(resource):
                tot = adj[index_shift + i][0][res]
                cmp = 0
                for k, sw in enumerate(sw_node):
                    cmp += adj[node_find[sw]][index_shift + i][res]
                if abs(tot - cmp) > 1E-7:
                    print("ERROR")
                else:
                    print("RIGHT")

    # 然后检查边的容量限制
    for res in range(resource):
        for k, sw in enumerate(sw_node):
            if adj[node_find[sw + "x"]][node_find[sw]][res] > capacity[sw][res] or \
                    adj[1][node_find[sw + "x"]][res] > capacity[sw][res] or \
                    adj[1][node_find[sw + "x"]][res] != adj[node_find[sw + "x"]][node_find[sw]][res]:
                print("ERROR")
            else:
                print(adj[1][node_find[sw + "x"]][res])
                print(adj[node_find[sw + "x"]][node_find[sw]][res])
                print("RIGHT")

    return adj, adj_copy


def update_adj(adj, s, t, usage, f, type1):
    #  1加 -1减
    for i in range(resource):
        if usage[i] > 0:
            adj[s][t][i] = adj[s][t][i] + type1 * usage[i] * f
            if abs(adj[s][t][i]) < 1E-7:
                adj[s][t][i] = 0


def dfs(adj, s, t, now, node_num, usage, pre, vis):
    if now == t:
        return True
    else:
        vis[now] = 1
        for i in range(1, node_num):
            if check_capacity(usage, adj[now][i]) and vis[i] == 0:
                pre[i] = now
                if dfs(adj, s, t, i, node_num, usage, pre, vis):
                    return True

        return False


def bfs(adj, s, t, node_num, usage, pre, vis):
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
