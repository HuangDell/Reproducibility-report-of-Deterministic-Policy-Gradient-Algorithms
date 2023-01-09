from test import *

if __name__ == '__main__':
    choose=int(input('1.mountain_car 2.pendulum 3.Octopus Arm\n'))
    if choose==1:
        test('MountainCarContinuous-v0')
    elif choose==2:
        test('PuddleWorld-v0')
    elif choose==3:
        test('Reacher-v4')





