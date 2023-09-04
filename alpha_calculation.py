import numpy as np
from scipy.linalg import eigh
from codes.utils import *
from codes.embedding import *
from codes.perturbation_attack import *
from codes.testModel import *
import time

def get_alpha(adj):
    deg = np.diag(adj.sum(1).A1)
    deno = 0
    dg = set()
    for i in range(len(deg)):
        di=deg[i,i]
        if(di<1):
            di=1
        dg.add(di)
        deno=deno+np.log(2*di)
    print(len(dg))
    print(deno)
    return 1+len(dg)*1/deno

_,adj,gul = getAdj(500)
adj = standardize(adj)
print(get_alpha(adj))

#dblp 1.0080174471890284
#wiki 1.0112308530067355