
import sys
sys.path.append('C:/Users/samib/Documents/AI/micrograd')
from micrograd.engine import Value 

# TEST EULER
a = Value(1.0)
b = a + a
c = b.euler()
print(f'grad before backpropagation {a.grad = :.4f}') # Initilized at 0
print(f'grad before backpropagation {b.grad = :.4f}') # Initilized at 0

print(f'{c.data:.4f}') # 7.3891
c.backward()

print(f'grad after backpropagation {a.grad = :.4f}') # a.grad = 14.7781
print(f'grad after backpropagation {b.grad = :.4f}') # b.grad = 7.3891 same as the value of c as in the derivative of exp will remain the same 

# TEST sigmoid

d = Value(2.0)
e = d.sigmoid()
print(f'{e.data:.4f}') 
e.backward()

print(f'grad after backpropagation for sigmoid function {d.grad:.4f}') # 0.1050

# TEST tanh

f = Value(2.0)
g = f.tanh()
print(f'{g.data:.4f}') 
g.backward()

print(f'grad after backpropagation for tanh function {f.grad:.4f}') # 0.0707