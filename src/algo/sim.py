from src.algo.base import *
from src.algo.params import *

# global resource
def one_task_one_resource():
    # global resource
    # resource = 1

    task1 = Task(0, [1])
    global tasks
    tasks.append(task1)
    base()


if __name__ == '__main__':
    # global resource
    # resource = 1
    one_task_one_resource()
