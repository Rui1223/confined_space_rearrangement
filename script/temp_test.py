#!/usr/bin/env python
from __future__ import division

import time
import sys
import os
import copy
import numpy as np

def gen(L, row, cur, res):
    if (row >= len(L)):
        # res.append(cur)
        if cur not in res:
            res.append(cur)
        return
    for col in range(len(L[row])):
        if L[row][col] not in cur:
            cur_new = cur+[L[row][col]]
            gen(L, row+1, cur_new, res)
        else:
            gen(L, row+1, cur, res)

if __name__ == '__main__':
    res = [] ### a list
    # L = [[1,2,3,4,7], [5,6,7,8,14], [7,8,6,9,12], [11,2,3,4,8], [4,5,6,1,9], [4,5,6,7,8], [15,16,17,18,19]]
    L = [[1,2,3,4], [5,6,7,8], [7,8,6,9], [11,2,3,4], [4,5,6,1], [4,5,6,7], [15,16,17,18]]
    start_time = time.time()
    gen(L, 0, [], res)
    print("time: " + str(time.time()-start_time))
    # print(res)
    print(len(res))
    


# class Father(object):
#     def __init__(self, a, b, c):
#         self.a = a
#         self.b = b
#         self.c = c
    
#     def add(self):
#         return self.a + self.b + self.c

# class QuietSon(Father):
#     def __init__(self, a, b, c):
#         Father.__init__(self, a, b, c)

# class NaughtrySon(Father):
#     def __init__(self, a, b, c):
#         Father.__init__(self, a, b, c)

#     def add(self):
#         return self.a * self.b * self.c

# if __name__ == '__main__':

#     quiet_son = QuietSon(2,3,4)
#     naughtry_son = NaughtrySon(2,3,4)

#     print(quiet_son.add())
#     print(naughtry_son.add())