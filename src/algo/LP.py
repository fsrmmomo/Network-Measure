import cplex
from cplex.exceptions import CplexError

from utils.util import *
from src.algo.base import Task, read_capaticy

tasks = []
capacity = dict()

# 资源种类数目
resource = 1

maxnum = 99999999


def algo():
    task1 = Task(0, [1])
    tasks.append(task1)
    global capacity
    # 得到sw的容量
    capacity = read_capaticy()

    LP_algo_continous()
    LP_algo_integer()


def LP_algo_continous():
    # 目标函数
    routing = get_routing()

    # 子流数目
    sub_flow_count = len(routing)

    # 任务数目
    tasks_count = len(tasks)

    # 交换机数目
    sws = get_sw_node_list()
    sw_count = len(sws)

    X = []
    my_ub = []
    # 流变量
    v_count = 0
    for i, task in enumerate(tasks):
        for j, r in enumerate(routing):
            # for sw in sws:
            for sw in r[:-1]:
                # for k in range(resource):
                name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                X.append(name)
                my_ub.append(r[-1])
                v_count += 1
                # print(name)

    # v_count = tasks_count*sub_flow_count*sw_count
    my_obj = [1 for _ in range(v_count)]
    my_lb = [0 for _ in range(v_count)]

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types="C" * v_count, names=X)

    # 首先是测量流量约束，每个任务中的每个子流分配在每个交换机的流总和不应该超出子流的大小
    rows = []
    my_rhs = []
    my_rownames = []
    for i, task in enumerate(tasks):
        for j, r in enumerate(routing):
            cons = []
            tmp = []
            for sw in r[:-1]:
                name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                tmp.append(name)
            cons.append(tmp)
            cons.append([1.0 for i in tmp])

            rows.append(cons)
            my_rhs.append(r[-1])
            my_rownames.append("task {0}-routing {1}constraint".format(i, j))

    # prob.linear_constraints.add(lin_expr=rows, senses="L" * len(rows),
    #                             rhs=my_rhs, names=my_rownames)
    #
    # # 然后是添加交换机节点容量约束，即每个交换机节点的每种资源限制都不能超过
    # rows = []
    # my_rhs = []
    # my_rownames = []
    for sw in sws:
        for res in range(resource):
            cons = []
            tmp_x = []
            tmp_a = []
            for i, task in enumerate(tasks):
                for j, r in enumerate(routing):
                    for sw1 in r[:-1]:
                        if sw1 == sw:
                            name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                            tmp_x.append(name)
                            tmp_a.append(task.usage[res])
            cons.append(tmp_x)
            cons.append(tmp_a)
            rows.append(cons)
            my_rhs.append(capacity[sw][res])
            my_rownames.append("sw {0}-res {1}".format(sw, res))
    prob.linear_constraints.add(lin_expr=rows, senses="L" * len(rows),
                                rhs=my_rhs, names=my_rownames)
    prob.linear_constraints.get_rows()
    try:
        prob.solve()
    except CplexError as exc:
        print(exc)
    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", prob.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print("Solution value  = ", prob.solution.get_objective_value())

    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()
    print(numcols)
    print(numrows)

    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()
    print(len(x))
    print(v_count)
    sw_count = dict()
    for sw in sws:
        sw_count[sw] = 0
    for i in range(len(x)):
        print(X[i], end=": ")
        print(x[i])

        sw = X[i].split("-")[-1].split(" ")[1]
        print(sw)
        sw_count[sw] = sw_count[sw] + x[i]
    print(sw_count)
    print(sum(x))
    #
    # print('x: ')
    # print(x)

def LP_algo_integer():

    mutil = 1E6
    # 目标函数
    routing = get_routing()

    # 子流数目
    sub_flow_count = len(routing)

    # 任务数目
    tasks_count = len(tasks)

    # 交换机数目
    sws = get_sw_node_list()
    sw_count = len(sws)

    X = []
    my_ub = []
    # 流变量
    v_count = 0
    for i, task in enumerate(tasks):
        for j, r in enumerate(routing):
            # for sw in sws:
            for sw in r[:-1]:
                # for k in range(resource):
                name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                X.append(name)
                my_ub.append(int(r[-1]*mutil))
                v_count += 1
                # print(name)

    # v_count = tasks_count*sub_flow_count*sw_count
    my_obj = [1 for _ in range(v_count)]
    my_lb = [0 for _ in range(v_count)]

    prob = cplex.Cplex()
    prob.objective.set_sense(prob.objective.sense.maximize)

    prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types="I" * v_count, names=X)

    # 首先是测量流量约束，每个任务中的每个子流分配在每个交换机的流总和不应该超出子流的大小
    rows = []
    my_rhs = []
    my_rownames = []
    for i, task in enumerate(tasks):
        for j, r in enumerate(routing):
            cons = []
            tmp = []
            for sw in r[:-1]:
                name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                tmp.append(name)
            cons.append(tmp)
            cons.append([1 for i in tmp])

            rows.append(cons)
            my_rhs.append(int(r[-1]*mutil))
            my_rownames.append("task {0}-routing {1}constraint".format(i, j))

    # prob.linear_constraints.add(lin_expr=rows, senses="L" * len(rows),
    #                             rhs=my_rhs, names=my_rownames)
    #
    # # 然后是添加交换机节点容量约束，即每个交换机节点的每种资源限制都不能超过
    # rows = []
    # my_rhs = []
    # my_rownames = []
    for sw in sws:
        for res in range(resource):
            cons = []
            tmp_x = []
            tmp_a = []
            for i, task in enumerate(tasks):
                for j, r in enumerate(routing):
                    for sw1 in r[:-1]:
                        if sw1 == sw:
                            name = "task {0}-routing {1}-sw {2}".format(i, j, sw)
                            tmp_x.append(name)
                            tmp_a.append(task.usage[res])
            cons.append(tmp_x)
            cons.append(tmp_a)
            rows.append(cons)
            my_rhs.append(int(capacity[sw][res]*mutil))
            my_rownames.append("sw {0}-res {1}".format(sw, res))
    prob.linear_constraints.add(lin_expr=rows, senses="L" * len(rows),
                                rhs=my_rhs, names=my_rownames)
    prob.linear_constraints.get_rows()
    try:
        prob.solve()
    except CplexError as exc:
        print(exc)
    print()
    # solution.get_status() returns an integer code
    print("Solution status = ", prob.solution.get_status(), ":", end=' ')
    # the following line prints the corresponding string
    print(prob.solution.status[prob.solution.get_status()])
    print("Solution value  = ", prob.solution.get_objective_value())

    numcols = prob.variables.get_num()
    numrows = prob.linear_constraints.get_num()
    print(numcols)
    print(numrows)

    slack = prob.solution.get_linear_slacks()
    x = prob.solution.get_values()
    print(len(x))
    print(v_count)
    sw_count = dict()
    for sw in sws:
        sw_count[sw] = 0
    for i in range(len(x)):
        print(X[i], end=": ")
        print(x[i])

        sw = X[i].split("-")[-1].split(" ")[1]
        print(sw)
        sw_count[sw] = sw_count[sw] + x[i]
    print(sw_count)
    print(sum(x))

if __name__ == '__main__':
    algo()
