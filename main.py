from test import *
from plot import *
if __name__ == '__main__':
    choose = int(input('1.mountain_car 2.pendulum 3.Octopus Arm\n'))
    if choose == 1:
        x, y, env_name = test('MountainCarContinuous-v0')
    elif choose == 2:
        x, y, env_name = test('Pendulum-v1')
    elif choose == 3:
        x, y, env_name = test('Reacher-v4')
    show(range(x), y,env_name)
