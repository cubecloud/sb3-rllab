from sb3_rllab.envtools.slocks import SMpLock, SThLock

if __name__ == "__main__":
    import multiprocessing
    import threading

    print('Mp rlock check test')
    obj1 = SMpLock(lock=multiprocessing.RLock(), unique_name='one')
    obj2 = SMpLock(lock=multiprocessing.RLock(), unique_name='two')
    obj3 = SMpLock(lock=multiprocessing.RLock(), unique_name='three')
    obj4 = SMpLock(lock=multiprocessing.RLock(), unique_name='three')

    print(id(obj1))  # prints the address of obj1
    print('Lock', id(obj1.lock))  # prints the address of obj1.lock
    print(id(obj2))  # prints the address of obj2
    print('Lock', id(obj2.lock))  # prints the address of obj1.lock
    print(id(obj3))  # prints the address of obj3
    print('Lock', id(obj3.lock))  # prints the address of obj1.lock
    assert obj1.lock != obj2.lock != obj3.lock, 'Lock objects cant be the same'
    print(id(obj4))  # prints the address of obj3
    print('Lock', id(obj4.lock))  # prints the address of obj1.lock
    assert obj3.lock == obj4.lock, 'Lock objects #3 and #4 must be the same'

    print(f'\nTh rlock check test')
    obj5 = SThLock(lock=threading.RLock(), unique_name='one')
    obj6 = SThLock(lock=threading.RLock(), unique_name='two')
    obj7 = SThLock(lock=threading.RLock(), unique_name='three')
    obj8 = SThLock(lock=threading.RLock(), unique_name='three')

    print(id(obj5))  # prints the address of obj1
    print('Lock', id(obj5.lock))  # prints the address of obj1.lock
    print(id(obj6))  # prints the address of obj2
    print('Lock', id(obj6.lock))  # prints the address of obj1.lock
    print(id(obj7))  # prints the address of obj3
    print('Lock', id(obj7.lock))  # prints the address of obj1.lock
    assert obj5.lock != obj6.lock != obj7.lock, 'Lock objects cant be the same'
    print(id(obj8))  # prints the address of obj3
    print('Lock', id(obj8.lock))  # prints the address of obj1.lock
    assert obj7.lock == obj8.lock, 'Lock objects #7 and #8 must be the same'
