from queue import Queue, Empty
from threading import Thread
from time import sleep
class TaskHandler:
    def __init__(self):
        self.__tasks = Queue()
        self.__worker = Thread(target=self.__process_tasks, daemon=True)
        self.__loop = True
        self.__worker.start()
    
    def add_task(self, func, *args, **kwargs):
        if 'repeat_after' in kwargs:
            Thread(target=self.__process_on_repeat, args=(func, args, kwargs), daemon=True).start()
        else:
            self.__tasks.put((func, args, kwargs))
    
    def __process_tasks(self):
        while self.__loop or not self.__tasks.empty():
            try:
                func, args, kwargs = self.__tasks.get(timeout=1)
                func(*args, **kwargs)
                self.__tasks.task_done()
            except Empty:
                continue
            except Exception as e:
                print(e)
    
    def __process_on_repeat(self, func, *args):
        repeat_after = args[1]['repeat_after']
        args = args[0]
        while self.__loop:
            func(*args)
            sleep(repeat_after)
    
    def stop(self):
        self.__loop = False
        self.__worker.join()