from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

import numpy as np

class Test:
    def __init__(self):
        self.indices = []

    def a(self, i):
        self.b = 1
        self.b = i

    def __call__(self):

        indices_to_add = np.arange(0,50,1)
        perrank = len(indices_to_add)//size

        low  = 0+rank*perrank
        high = 0+(rank+1)*perrank
        if rank == size-1:
            high = len(indices_to_add)
        
        for i in range(low, high):
            self.a(i)
            self.indices.append(self.b)

t = Test()
t()

all_indices = comm.gather(t.indices, root=0)
print(rank, all_indices)
exit()

#comm.Barrier()
all_indices = []
for rank_i in range(size):
    if rank == rank_i:
        all_indices.append(t.indices)

if rank == 0:
    print(t.indices)
    print(all_indices)
if rank == 1:
    print(t.indices)
    print(all_indices)