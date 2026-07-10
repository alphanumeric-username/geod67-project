from geod67lib import fd

import time

def timeit(f):
    t0 = time.time()
    f()
    return time.time() - t0


print(timeit(lambda : fd.taylor_coeffs(16)))
print(timeit(lambda : fd.taylor_coeffs(16)))
print(timeit(lambda : fd.taylor_coeffs(16)))
print(timeit(lambda : fd.taylor_coeffs(16)))
print(timeit(lambda : fd.taylor_coeffs(16)))
print(timeit(lambda : fd.taylor_coeffs(16)))