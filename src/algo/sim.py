from src.algo.base import *


def one_task_one_resource():
    global resource
    resource = 1

    task1 = Task(0, [1])
    global tasks
    tasks.append(task1)
    base()

if __name__ == '__main__':
    one_task_one_resource()