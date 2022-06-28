import torch as t
import numpy as np

b = 50
c = 128
d = 60

x = t.rand(b,c,d).to('cuda')
y = t.rand(b,c,d).to('cuda')

print(x.shape, y.shape)
z = t.bmm(x.permute(0,2,1), y)
import pdb; pdb.set_trace()

print(z)
