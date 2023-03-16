import os

from numpy import mean

from utils.util import *
from queue import Queue
import random
from src.algo.params import *
from src.algo.LP import *


# tasks = []
# capacity = dict()
# resource = 1
# maxnum = 99999999

# global tasks
# global capacity
# global resource
# global maxnum

class Task:
    def __init__(self, ID, usage):
        self.ID = ID
        self.usage = usage
        self.weight = 0
        self.remain = 0
        self.remain_list = []
        self.remain_index = 0


def read_capacity():
    clist = dict()
    with open("../../data/capacity", "r") as f:
        for line in f.readlines():
            sw = line.split()[0]
            c = []
            for i in range(1):
                c.append(int(line.split()[i + 1]))
                c.append(1000000)
            clist[sw] = c
    return clist


def set_capacity(clist, new_list):
    for i, c in enumerate(clist.keys()):
        clist[c] = new_list[i]


def base(routing):
    return algo(capacity, routing)


def algo(capacity, routing):
    # for t in tasks:
    #     t.remain_list = [i for i in range(len(routing))]
    #     print(t.remain_list)
    #     random.shuffle(t.remain_list)
    #     print(t.remain_list)

    # print(sum([i[-1] for i in routing]))
    task_num = len(tasks)
    for t in tasks:
        t.remain = len(routing)
    sw_node = get_sw_node_list()
    node_num = 1 + 1 + len(sw_node) * 2 + task_num * len(routing)

    # 创建图 adj_copy用于对比
    # 按照优先级排序
    # sort_res = sort_task_and_subflow(routing)

    # sort_res = sort_task_and_subflow(routing)
    sort_res = sort_task_only(routing)
    adj, adj_copy= build_graph(capacity, task_num, node_num, sw_node, routing)

    # task_num = 1
    # node_num = 1 + 1 + len(sw_node) * 2 + task_num * len(routing)
    # adj, adj_copy = build_graph(capacity, task_num, node_num, sw_node, routing)

    sum1 = 0
    sum2 = 0
    for i in range(node_num):
        for j in range(node_num):
            sum1 += sum(adj[i][j])
            sum2 += sum(adj_copy[i][j])
    # print(sum1)
    # print(sum2)
    # print(adj)
    # print(adj_copy)

    ans = max_flow(adj, task_num, node_num, sw_node, routing, sort_res)

    # check_result(adj, adj_copy, task_num, node_num, sw_node, routing)
    # get_ans(adj, adj_copy, task_num, node_num, sw_node, routing)
    print(ans)
    return get_ans(adj, adj_copy, task_num, node_num, sw_node, routing)


def build_graph(capacity, task_num, node_num, sw_node, routing):
    # print(node_num)
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

    # print(len(adj))
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


def max_flow(adj, task_num, node_num, sw_node, routing, sort_res):
    # 从start相邻的每个节点开始，使用BFS进行计算
    f_each_task = len(routing)
    ans = 0

    for pair in sort_res:
        ID = pair[0]
        routing_index = pair[1]
        # 添加start到subflow的边
        # adj[0][index] = r[-1]
        r = routing[routing_index]
        f = 2 + len(sw_node) * 2 + ID * f_each_task + routing_index
        res = r[-1]
        task_usage = tasks[ID].usage
        while res > 1E-7:
            pre = [0 for _ in range(node_num)]
            vis = [0 for i in range(node_num)]
            # if bfs(adj, f, 1, node_num, task_usage, pre, vis):
            vis[0] = 1
            pre[f] = 0

            # def bfs(adj, s, t, node_num, usage, pre, vis):
            if bfs(adj, f, 1, node_num, task_usage, pre, vis):

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
                # print(mind)
                # print(res)
                # print(ans)
                now = 1
                while now != 0:
                    last = pre[now]
                    update_adj(adj, last, now, task_usage, mind, -1)
                    update_adj(adj, now, last, task_usage, mind, 1)
                    now = last
                # print()
            else:
                break

    # 从优先级高的任务开始 bfs
    # print(tasks)
    for i, t in enumerate(tasks):
        print(i)
        # print("""\n\n\n\n在这里开始\n\n\n""")
        # print(t)
        # 每个任务遍历
        index_start = 2 + len(sw_node) * 2 + i * f_each_task
        task_usage = t.usage
        for f in range(index_start, index_start + f_each_task):
            # if adj[0][f]
            res = routing[f - index_start][-1]
            res = min_capacity(adj[0][f], res, task_usage)
            while res > 1E-7:
                pre = [0 for _ in range(node_num)]
                vis = [0 for i in range(node_num)]
                # if bfs(adj, f, 1, node_num, task_usage, pre, vis):
                vis[0] = 1
                pre[f] = 0
                # if dfs(adj, f, 1, f, node_num, task_usage, pre, vis):
                if bfs(adj, f, 1, node_num, task_usage, pre, vis):
                    # print("""\n\n\n\n在这里找到了\n\n\n""")

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
                    # print(mind)
                    # print(res)
                    # print(ans)
                    now = 1
                    while now != 0:
                        last = pre[now]
                        update_adj(adj, last, now, task_usage, mind, -1)
                        update_adj(adj, now, last, task_usage, mind, 1)
                        now = last
                    # print()
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
    # print('111')
    return ans


def get_ans(adj, adj_copy, task_num, node_num, sw_node, routing):
    # print(node_num)

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
    # sort_task_and_subflow()
    for taskID in range(task_num):
        for i, r in enumerate(routing):
            name = "task " + str(taskID) + " subflow " + str(i)
            node_map.append(name)
            node_find[name] = index
            index += 1

    # 首先检查分流量限制是否满足

    index_shift = 42
    ans = 0
    ans2 = 0
    f_each_task = len(routing)
    for taskID in range(task_num):
        for i, r in enumerate(routing):
            # for res in range(resource):
            f = index_shift + i + f_each_task*taskID
            # tot = adj[index_shift + i][0][0]
            # ans2 += adj[index_shift + i][0][0] / tasks[taskID].usage[0]
            # for k, sw in enumerate(sw_node):
            #     ans += int(adj[node_find[sw]][index_shift + i][0] / tasks[taskID].usage[0])
            # tot = adj[f][0][0]
            # ans2 += adj[f][0][0] / tasks[taskID].usage[0]
            for k, sw in enumerate(sw_node):
                ans += int(adj[node_find[sw]][f][0] / tasks[taskID].usage[0])
            # ans += int(tot / tasks[taskID].usage[0])
    print("实际测量的流的数量为：" + str(ans))
    # print("实际测量的流的数量2为：" + str(ans2))
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
    # sort_task_and_subflow()
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
        # print(resource)
        # print(i)
        if usage[i] > 0 and cap[i] <= 0:
            return False
    return True


def min_capacity(cap, mind, usage):
    for i in range(resource):
        if usage[i] > 0:
            mind = min(mind, cap[i] / usage[i])
    return mind


# 实际上我们只需要排序task 因为同一个task里面的任务肯定能够最大化，不需要排序
def sort_task_and_subflow_random(routing):
    if len(tasks) == 1:
        result = []
        for i in range(len(routing)):
            result.append([0, i])
        return result
    # total_c = [sum([capacity[j][i] for j in range(len(capacity))]) for i in range(resource)]
    total_c = [sum([capacity[j][i] for j in capacity.keys()]) for i in range(resource)]
    result = []
    task_box = [task for task in tasks]

    while len(task_box) > 0:
        tmp_tasks = tasks[:]
        weight = [0 for _ in range(len(task_box))]
        for j in range(len(task_box)):
            for i in range(resource):
                weight[j] += task_box[j].usage[i] / total_c[i]
            task_box[j].weight = weight[j]
        task_box.sort(key=lambda x: x.weight)
        # result.append(task_box[0].ID)
        result.append([task_box[0].ID, task_box[0].remain_list[task_box[0].remain_index]])

        for i in range(resource):
            # total_c[i] -= task_box[0].usage[i] * routing[(len(routing) - task_box[0].remain)][-1]
            total_c[i] -= task_box[0].usage[i] * routing[task_box[0].remain_list[task_box[0].remain_index]][-1]
            if total_c[i] < 1E-7:
                total_c[i] = 0
        task_box[0].remain -= 1
        task_box[0].remain_index += 1
        print(task_box[0].ID)
        print(total_c)
        # print()
        # 删除没有流的和没有空间容量的
        tmp = []
        for t in task_box:
            if t.remain == 0:
                continue
            else:
                flag = True
                for i in range(resource):
                    if t.usage[i] != 0 and total_c[i] == 0:
                        flag = False
                        break
                if flag:
                    tmp.append(t)
        task_box = tmp

    print(result)
    return result

    # tasks.sort(key=lambda x: x.weight)


def sort_task_only(routing):
    if len(tasks) == 1:
        return []
    total_c = [sum([capacity[j][i] for j in capacity.keys()]) for i in range(resource)]
    # task_box = [task for task in tasks]
    weight = [0 for _ in range(len(tasks))]
    for j in range(len(tasks)):
        for i in range(resource):
            weight[j] += tasks[j].usage[i] / total_c[i]
        tasks[j].weight = weight[j]
    tasks.sort(key=lambda x: x.weight)
    return []


def sort_task_and_subflow(routing):
    if len(tasks) == 1:
        result = []
        for i in range(len(routing)):
            result.append([0, i])
        return result
    # total_c = [sum([capacity[j][i] for j in range(len(capacity))]) for i in range(resource)]
    total_c = [sum([capacity[j][i] for j in capacity.keys()]) for i in range(resource)]
    result = []
    task_box = [task for task in tasks]
    while len(task_box) > 0:
        weight = [0 for _ in range(len(task_box))]
        for j in range(len(task_box)):
            for i in range(resource):
                if total_c[i] != 0:
                    weight[j] += task_box[j].usage[i] / total_c[i]
            task_box[j].weight = weight[j]
        task_box.sort(key=lambda x: x.weight)
        # result.append(task_box[0].ID)
        result.append([task_box[0].ID, len(routing) - task_box[0].remain])

        for i in range(resource):
            total_c[i] -= task_box[0].usage[i] * routing[(len(routing) - task_box[0].remain)][-1]
            # total_c[i] -= task_box[0].usage[i] * routing[task_box[0].remain_list[task_box[0].remain_index]][-1]
            if total_c[i] < 1E-7:
                total_c[i] = 0
        task_box[0].remain -= 1
        task_box[0].remain_index += 1
        print(task_box[0].ID)
        print(total_c)
        # print()
        # 删除没有流的和没有空间容量的
        tmp = []
        for t in task_box:
            if t.remain == 0:
                continue
            else:
                flag = True
                for i in range(resource):
                    if t.usage[i] != 0 and total_c[i] == 0:
                        flag = False
                        break
                if flag:
                    tmp.append(t)
        task_box = tmp
    count_dict = dict()
    sum_f = 0
    for r in result:
        # sum_f += 1
        # if r[0] in count_dict.keys():
        #     count_dict[r[0]] += 1
        # else:
        #     count_dict[r[0]] = 1
        sum_f += routing[r[1]][-1]
        if r[0] in count_dict.keys():
            count_dict[r[0]] += routing[r[1]][-1]
        else:
            count_dict[r[0]] = routing[r[1]][-1]

    new_usage = [0 for i in range(resource)]
    for k, v in count_dict.items():
        print(tasks[k].usage)
        for i in range(resource):
            new_usage[i] += tasks[k].usage[i] * (v / sum_f)
    new_task = Task(0, new_usage)
    # global tasks
    tasks.clear()
    tasks.append(new_task)
    multi = sum_f / max(count_dict.values())
    print(multi)
    print(new_usage)
    for i, r in enumerate(routing):
        routing[i][-1] *= multi
    print(routing)
    print(count_dict)
    print(result)

    return []


def set_routing(routing, new_fd_list):
    for i, r in enumerate(routing):
        r[-1] = new_fd_list[i]





def get_step_clist_same_node(tot=20800000):
    if os.path.isfile("../../data/same_c.pkl"):
        # pass
        return pkl_read("../../data/same_c.pkl")
    sw_num = 20
    res_num = 10
    multi = 0.2
    res = [[[int(tot / 20 * step * multi)] for _ in range(sw_num)] for step in range(1, 11)]
    pkl_write("../../data/same_c.pkl", res)
    print(res)
    return res

def get_step_clist_diff_node(tot=20800000):
    if os.path.isfile("../../data/diff_c.pkl"):
        # pass
        return pkl_read("../../data/diff_c.pkl")
    multi = 0.2
    tot_list = [int(tot)*step*multi for step in range(1, 11)]
    # 每个容量下生成的拓扑数量
    topo_num = 100
    res = []
    tmp = []
    aa = 1000
    # 每个节点 20-100之间
    plist = []
    for i in range(100):
        while True:
            add = [random.randint(20,80) for _ in range(20)]
            s = aa / sum(add)
            add = [a*s for a in add]
            if min(add)>=20 and max(add)<=200:

                add = [a/aa for a in add]
                plist.append(add)
                break
    print(plist)
    res = []
    for t in tot_list:
        tmp = []
        for i,p in enumerate(plist):
            # plist[i] = [a*t for a in p]
            tmp.append([[int(a*t)] for a in p])
        print(mean(tmp[-1]))
        print(tmp)
        res.append(tmp)

    pkl_write("../../data/diff_c.pkl", res)
    print(res)
    return res

def get_same_f(tot=20800000):
    return [int(1E5) for _ in range(208)]

def get_diff_f(tot=20800000):
    if os.path.isfile("../../data/diff_f.pkl"):
        return pkl_read("../../data/diff_f.pkl")
    plist = []
    for i in range(100):
        while True:
            add = [random.randint(30000,170000) for _ in range(208)]
            s = tot / sum(add)
            add = [int(a*s) for a in add]
            # print(add)
            if min(add)>=20000 and max(add)<=2E5:

                # add = [a/aa for a in add]
                print(add)
                print(mean(add))
                plist.append(add)
                break
    print(plist)

    pkl_write("../../data/diff_f.pkl", plist)
    return plist

def get_dif_task(n,r):
    tt = []
    for i in range(r):
        tlist = [random.randint(1,10) for _ in range(n)]
        ss = sum(tlist)
        tlist = [t/ss for t in tlist]
        print(tlist)
        tt.append(tlist)
    res = []
    for i in range(n):
        task = Task(i, [tlist[i]  for tlist in tt])
        res.append(task)
    return res


def get_task_file():
    if os.path.isfile("../../data/task_file.pkl"):
        return pkl_read("../../data/task_file.pkl")
    t = [[[] for i in range(0,11)] for j in range(0,11)]
    for i in range(1,11):
        for j in range(1,11):
            tmp = []
            for k in range(100):
                tmp.append(get_dif_task(i,j))
            t[i][j] = tmp
            print(len(t[i][j]))
    print(t)
    pkl_write("../../data/task_file.pkl",t)
    return t



def one_task_one_resource(mode="1000"):
    global resource
    resource = 1
    task1 = Task(0, [1])
    global tasks
    tasks.clear()
    tasks.append(task1)

    global capacity
    # 得到sw的容量
    capacity = read_capacity()

    new_clist = [[1E6 + 120000 for _ in range(resource)] for _ in range(len(capacity))]
    set_capacity(capacity, new_clist)


    routing = get_routing()
    new_fd_list = [1E5 for _ in range(len(routing))]
    set_routing(routing, new_fd_list)

    # c_ans = base(routing)
    # lp_ans = LP_algo_integer(routing, capacity, resource)
    #
    # print(c_ans / lp_ans)

    # 使用相同的流量大小，相同的节点容量，根据不同的C比例计算结果
    if mode[0] == '1':
        result_1 = []
        new_clist = get_step_clist_same_node()
        new_fd_list = get_same_f()
        for i in range(10):
            set_capacity(capacity,new_clist[i])
            set_routing(routing, new_fd_list)
            c_ans = base(routing)
            lp_ans = LP_algo_integer(routing, capacity, resource)
            print(c_ans)
            print(lp_ans)
            print(c_ans / lp_ans)
            result_1.append([c_ans,lp_ans,c_ans / lp_ans])
        print(result_1)
        with open("../../result/ss-1","w") as f:
            for r in result_1:
                # f.writelines(r)
                f.write(str(r))
                f.write('\n')


    # 使用相同的流量大小，不同的节点容量，根据不同的C比例计算结果
    if mode[1] == '1':
        result_2 = []
        new_clist = get_step_clist_diff_node()
        new_fd_list = get_same_f()
        # print(new_clist)
        # print(new_fd_list)
        for i in range(10):
            tmp = []
            for j in range(100):
                set_capacity(capacity,new_clist[i][j])
                set_routing(routing, new_fd_list)
                c_ans = base(routing)
                lp_ans = LP_algo_integer(routing, capacity, resource)
                print(c_ans)
                print(lp_ans)
                print(c_ans / lp_ans)
                tmp.append(c_ans / lp_ans)
            result_2.append(tmp)
        print(result_2)
        with open("../../result/ss-2","w") as f:
            for r in result_2:
                # f.writelines(r)
                f.write(str(r))
                f.write('\n')


    # 使用不同的流量大小，相同的节点容量，根据不同的C比例计算结果
    result_3 = []
    if mode[2] == '1':
        print("第三种")
        result = []
        new_clist = get_step_clist_same_node()
        new_fd_list = get_diff_f()
        # print(new_clist)
        # print(new_fd_list)
        for i in range(1):
            tmp = []
            for j in range(10):
                set_capacity(capacity,new_clist[i])
                set_routing(routing, new_fd_list[j])
                c_ans = base(routing)
                lp_ans = LP_algo_integer(routing, capacity, resource)
                print(c_ans)
                print(lp_ans)
                print(c_ans / lp_ans)
                tmp.append(c_ans / lp_ans)
            result.append(tmp)
        print(result)
        with open("../../result/ss-3","w") as f:
            for r in result:
                # f.writelines(r)
                f.write(str(r))
                f.write('\n')


    # 使用不同的流量大小，不同的节点容量，根据不同的C比例计算结果
    if mode[3]=='1':

        print("第4种")
        result = []
        new_clist = get_step_clist_diff_node()
        new_fd_list = get_diff_f()
        for i in range(1):
            tmp = []
            for j in range(10):
                set_capacity(capacity,new_clist[i][j])
                set_routing(routing, new_fd_list[0])
                c_ans = base(routing)
                lp_ans = LP_algo_integer(routing, capacity, resource)
                print(c_ans)
                print(lp_ans)
                print(c_ans / lp_ans)
                tmp.append(c_ans / lp_ans)
            result.append(tmp)
        print(result)
        with open("../../result/ss-4","w") as f:
            for r in result:
                # f.writelines(r)
                f.write(str(r))
                f.write('\n')

def one_task_m_resource(mode="0000"):
    global resource
    resource = 1
    task1 = Task(0, [2])
    global tasks
    tasks.clear()
    tasks.append(task1)

    global capacity
    # 得到sw的容量
    capacity = read_capacity()

    new_clist = [[1E6 + 120000 for _ in range(resource)] for _ in range(len(capacity))]
    set_capacity(capacity, new_clist)


    routing = get_routing()
    new_fd_list = [1E5 for _ in range(len(routing))]
    set_routing(routing, new_fd_list)

    # 第一个仿真，根据使用资源数量
def m_task_one_resource(mode="10000"):
    global resource
    resource = 1
    # task1 = Task(0, [1])
    global tasks
    tasks.clear()
    # tasks.append(task1)
    global capacity
    # 得到sw的容量
    capacity = read_capacity()
    routing = get_routing()

    tt = get_task_file()


    # 不同任务数量的结果,从1-10,资源为1,各自随机生成100个组合
    if mode[0] == "1":
        result = []
        new_clist = get_step_clist_same_node()
        new_fd_list = get_same_f()
        #
        for i in range(10):
            # 流和c不变
            set_capacity(capacity, new_clist[2]) # 使用总资源0.6比例，更好观察效果
            set_routing(routing, new_fd_list)
            tmp = []
            for j in range(5):
                new_tasks = tt[i+1][1][j]
                tasks.clear()
                tasks.extend(new_tasks)
                # c_ans = base(routing)
                lp_ans = LP_algo_integer(routing, capacity, resource)
                print(lp_ans)
                tmp.append(lp_ans)
                # print(c_ans)
                # print(lp_ans)
                # print(c_ans / lp_ans)
                # tmp.append(c_ans)
                # tmp.append(c_ans / lp_ans)
            result.append(tmp)
        print(result)
        plain_save("../../result/ms-12", result)

    # 4个任务下，资源为1，f相同，c相同，资源占比改变的计算结果
    if mode[1] == "1":
        result = []
        new_clist = get_step_clist_same_node()
        new_fd_list = get_same_f()
        #
        set_routing(routing, new_fd_list)
        for i in range(10):
            # 流和c不变i
            set_capacity(capacity, new_clist[i]) #
            tmp = []
            for j in range(5):
                new_tasks = tt[4][1][j]
                tasks.clear()
                tasks.extend(new_tasks)
                tmp.append(compute(new_clist[i],new_fd_list,resource,routing))
            result.append(tmp)
        print(result)
        plain_save("../../result/ms-22",result)

    # 4个任务下，资源为1，f相同，c不相同，资源占比改变的计算结果
    if mode[2] == "1":
        result = []
        new_clist = get_step_clist_diff_node()
        new_fd_list = get_same_f()
        for i in range(10):
            # 流和c不变i
            tmp = []
            # k对应不同容量分布
            for k in range(1):
                # set_capacity(capacity, new_clist[i]) #
                # set_routing(routing, new_fd_list)
                # j 代表不同的任务序列
                for j in range(5):
                    new_tasks = tt[4][1][j]
                    tasks.clear()
                    tasks.extend(new_tasks)
                    tmp.append(compute(new_clist[i][k],new_fd_list,resource,routing))
                    #
                    # c_ans = base(routing)
                    # lp_ans = LP_algo_integer(routing, capacity, resource)
                    # print(c_ans)
                    # print(lp_ans)
                    # print(c_ans / lp_ans)
                    # tmp.append(c_ans / lp_ans)

            result.append(tmp)
        print(result)
        plain_save("../../result/ms-32", result)


    # 4个任务下，资源为1，f不同，c相同，资源占比改变的计算结果
    if mode[3] == "1":
        result = []
        new_clist = get_step_clist_same_node()
        new_fd_list = get_diff_f()
        for i in range(10):
            # 流和c不变i
            tmp = []
            # k对应不同流量分布
            for k in range(1):
                # j 代表不同的任务序列
                for j in range(5):
                    new_tasks = tt[4][1][j]
                    tasks.clear()
                    tasks.extend(new_tasks)
                    tmp.append(compute(new_clist[i],new_fd_list[k],resource,routing))

            result.append(tmp)
        print(result)
        plain_save("../../result/ms-42",result)

    # 4个任务下，资源为1，f不同，c不同，资源占比改变的计算结果
    if mode[4] == "1":
        result = []
        new_clist = get_step_clist_diff_node()
        new_fd_list = get_diff_f()
        for i in range(10):
            # 流和c不变i
            tmp = []
            # k对应不同流量分布
            for k in range(1):
                # j 代表不同的任务序列
                for j in range(5):
                    new_tasks = tt[4][1][j]
                    tasks.clear()
                    tasks.extend(new_tasks)
                    tmp.append(compute(new_clist[i][2],new_fd_list[k],resource,routing))

            result.append(tmp)
        print(result)
        plain_save("../../result/ms-52",result)

def plain_save(filename,result):
    with open(filename, "w") as f:
        for r in result:
            f.write(str(r))
            f.write('\n')

def compute(new_clist,new_fd_list,resource,routing):
    global capacity
    set_capacity(capacity, new_clist)
    set_routing(routing,new_fd_list)
    c_ans = base(routing)
    # lp_ans = LP_algo_integer(routing, capacity, resource)
    print(c_ans)
    return c_ans
    # print(lp_ans)
    # print(c_ans / lp_ans)
    # tmp.append(c_ans / lp_ans)
    # return c_ans / lp_ans


if __name__ == '__main__':
    # global resource
    # resource = 1
    # one_task_one_resource()
    # get_step_clist_same_node()
    # get_step_clist_diff_node(tot=20800000)
    # get_diff_f()
    # get_dif_task(5,3)
    # get_task_file()
    m_task_one_resource()

