import pyopencl as cl
import pyopencl.array
import numpy as np

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n = 80 * (2**10)**2

mf = cl.mem_flags
a = cl.array.zeros(queue, n, np.float32)

prg = cl.Program(ctx, """//CL//
    kernel void fill_vec(global float *a, int chunk_size)
    {
      int i = get_global_id(0);

      float val_0 = a[i];

      a[i] = 2*val_0;
    }

    """).build()

from time import time

ntrips = 10

queue.finish()
t1 = time()
for i in xrange(ntrips):
    prg.fill_vec(queue, (n,), (256,), a.data, np.int32(n))
queue.finish()
t2 = time()
print "GB / sec: %g" % (2*a.nbytes*ntrips/(t2-t1)*1e-9)

# vim: filetype=pyopencl
