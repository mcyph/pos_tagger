from _thread import allocate_lock

GLOBAL_LOCK = allocate_lock()
DInit = {}


class Singleton(object):
    def __new__(cls, *args, **kwargs):
        """
        Only ever store a single instance in memory (singleton pattern)
        """

        # Technically, this is closer to a "borg" as proposed by Alex Martelli
        # (https://www.oreilly.com/library/view/python-cookbook/0596001673/ch05s23.html)
        # than an actual singleton implementation.

        # It's surprisingly difficult to allow for singletons which are both clean
        # in code and don't introduce side effects, such as if two of the same
        # class are being created at exactly the same time, or dealing with the
        # delay between when __new__ and __init__ is called. This uses both
        # global and local locks to try to prevent this.

        with GLOBAL_LOCK:
            if not hasattr(cls, '_state'):
                #print("CREATING SINGLETON FOR:", cls)
                cls._state = {}
                init_run = [False]
                old_init = cls.__init__
                local_lock = allocate_lock()

                def init(self, *args, **kw):
                    with local_lock: # Make sure initial init
                                     # can't happen at same time!
                        if not init_run[0]:
                            old_init(self, *args, **kw)
                            init_run[0] = True

                cls.__init__ = init

        self = object.__new__(cls)
        self.__dict__ = cls._state
        return self


if __name__ == '__main__':
    class TestClass(Singleton):
        def __init__(self):
            print("Should only run once!")
            self.value = 'blah'

        def check(self):
            self.value
            #print(self.value)

    import _thread

    def fn():
        for x in range(500000):
            a = TestClass()
            b = TestClass()
            #assert(a == b)
            a.check()
            b.check()

    for x in range(500):
        _thread.start_new(fn, ())
    while 1: pass
