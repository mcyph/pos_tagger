import os
import signal
import atexit
from _thread import allocate_lock
from multiprocessing import Process, Queue


def _engine_process(cls, cmd_q, response_q, iso, use_gpu):
    cls = cls(iso, use_gpu)

    while True:
        try:
            cmd, args, kw = cmd_q.get()
            if cmd == 'exit':
                raise SystemExit
            response_q.put(('ok', getattr(cls, cmd)(*args, **kw)))
        except SystemExit:
            import traceback
            traceback.print_exc()
            raise
        except Exception as e:
            response_q.put(('error', repr(e)))
            import traceback
            traceback.print_exc()


class EngineProcess:
    def __init__(self, cls, iso, use_gpu=False):
        """
        Newer machine learning-based POS taggers unfortunately seem to
        interfere with each other a lot, especially when using the GPU.

        While this isn't the most wonderful solution, the initialization
        times and memory usage for each engine is normally quite high,
        so the memory and CPU overhead of having things in a separate
        process is the best solution I can currently come up with.
        """
        self.lock = allocate_lock()
        cmd_q = self.cmd_q = Queue()
        response_q = self.response_q = Queue()
        self.p = Process(
            target=_engine_process,
            args=(cls, cmd_q, response_q, iso, use_gpu)
        )
        self.p.start()
        atexit.register(self.__del__)

    def __del__(self):
        with self.lock:
            if self.p:
                atexit.unregister(self.__del__)
                os.kill(self.p.pid, signal.SIGTERM)
                self.cmd_q.put(('exit', None, None))
                self.p.join()
                self.p = None

    def get_L_sentences(self, s):
        with self.lock:
            self.cmd_q.put(('get_L_sentences', (s,), {}))
            response_typ, data = self.response_q.get()
            if response_typ == 'error':
                raise Exception(data)
            else:
                assert response_typ == 'ok'
                return data
