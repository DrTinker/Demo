from threading import Thread
from queue import Queue


class ThreadGenerator(object):

    def __init__(self, iterator,
                 sentinel=object(),
                 queue_maxsize=0,
                 deamon=False):
        self._iterator = iterator
        self._sentinel = sentinel
        self._queue = Queue(maxsize=queue_maxsize)
        self._thread = Thread(
            name=repr(iterator),
            target=self._run
        )
        self._thread.deamon = deamon
        self._start = False

    def __repr__(self):
        return 'ThreadGenerator({!r})'.format(self._iterator)

    def _run(self):
        try:
            for value in self._iterator:
                if not self._start:
                    return
                self._queue.put(value)
        finally:
            self._queue.put(self._sentinel)

    def close(self, time=30):
        self._start = False
        try:
            while True:
                self._queue.get(timeout=time)
        except KeyboardInterrupt:
            print('手动终止')
        finally:
            pass

    def __iter__(self):
        self._start = True
        self._thread.start()
        for value in iter(self._queue.get, self._sentinel):
            yield value
        self._thread.join()
        self._start = False

    def __next__(self, time=30):
        if not self._start:
            self._start = True
            self._thread.start()
        value = self._queue.get(timeout=time)
        if value == self._sentinel:
            print('生成结束')
            # raise StopIteration()
        return value


def test():
    def gene():
        i = 0
        while (i < 2):
            print('aa')
            # yield i
            print('走了')
            i += 1

    t1 = gene()
    # t2 = gene()
    test = ThreadGenerator(t1)
    # test1 = ThreadGenerator(t2)

    for i in t1:
        print(next(test))

    test.close()
    # test1.close()


if __name__ == '__main__':
    test()
