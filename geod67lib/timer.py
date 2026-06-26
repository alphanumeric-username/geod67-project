import time

class Timer:
    def __init__(self):
        self.t0 = None

    def start(self):
        self.t0 = time.time()
    
    def stop(self):
        if not(self.t0 is None):
            tn = time.time()
            dt = tn - self.t0
            self.t0 = None
            return dt
        return -1