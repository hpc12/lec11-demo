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
        case 0: a[i] = 32; break;
        case 1: a[i] = 0; break;
        case 2: a[i] = 12; break;
        case 3: a[i] = 1; break;
        case 4: a[i] = 0; break;
        case 5: a[i] = 5; break;
        case 6: a[i] = 7; break;
        case 7: a[i] = 9; break;
        case 8: a[i] = 5; break;
        case 9: a[i] = 3; break;
        case 10: a[i] = 59; break;
        case 11: a[i] = 4; break;
        case 12: a[i] = 13; break;
        case 13: a[i] = -3; break;
        case 14: a[i] = 5; break;
        case 15: a[i] = 0; break;
        case 16: a[i] = 4; break;
        case 17: a[i] = -11; break;
        case 18: a[i] = 9; break;
        case 19: a[i] = 999; break;
        case 20: a[i] = 332; break;
        case 21: a[i] = 30; break;
        case 22: a[i] = 312; break;
        case 23: a[i] = 31; break;
        case 24: a[i] = 30; break;
        case 25: a[i] = 35; break;
        case 26: a[i] = 37; break;
        case 27: a[i] = 39; break;
        case 28: a[i] = 31; break;
        case 29: a[i] = 33; break;
        case 30: a[i] = 11111; break;
        case 31: a[i] = -33; break;
        default: ; break;
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

