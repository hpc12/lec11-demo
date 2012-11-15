import pyopencl as cl
import pyopencl.array
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 80 * (2**10)**2

mf = cl.mem_flags
a = cl.array.zeros(queue, n, np.float32)

prg = cl.Program(ctx, """//CL//
    kernel void choices(global float *a)
    {
      int i = get_global_id(0);
      switch (i % 32)
      {
        case 0: a[i] = 32;
        case 1: a[i] = 0;
        case 2: a[i] = 12;
        case 3: a[i] = 1;
        case 4: a[i] = 0;
        case 5: a[i] = 5;
        case 6: a[i] = 7;
        case 7: a[i] = 9;
        case 8: a[i] = 5;
        case 9: a[i] = 3;
        case 10: a[i] = 59;
        case 11: a[i] = 4;
        case 12: a[i] = 13;
        case 13: a[i] = -3;
        case 14: a[i] = 5;
        case 15: a[i] = 0;
        case 16: a[i] = 4;
        case 17: a[i] = -11;
        case 18: a[i] = 9;
        case 19: a[i] = 999;
        case 20: a[i] = 332;
        case 21: a[i] = 30;
        case 22: a[i] = 312;
        case 23: a[i] = 31;
        case 24: a[i] = 30;
        case 25: a[i] = 35;
        case 26: a[i] = 37;
        case 27: a[i] = 39;
        case 28: a[i] = 31;
        case 29: a[i] = 33;
        case 30: a[i] = 11111;
        case 31: a[i] = -33;
        default: ;
      }
    }

    """).build()

from time import time

ntrips = 10

prg.choices(queue, a.shape, (256,), a.data)

queue.finish()
t1 = time()
for i in xrange(ntrips):
    prg.choices(queue, a.shape, (256,), a.data)
queue.finish()
t2 = time()
print "M entries per sec: %g" % (ntrips*n/(t2-t1)*1e-6)

# vim: filetype=pyopencl

