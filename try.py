import numpy as np
class test:
  def __init__(self):
    self.a=np.arange(10)
  def __call__(self):
    np.random.shuffle(self.a)
    for i in self.a:
      yield i

def test1():
  a = np.arange(5)
  np.random.shuffle(a)
  for i in a:
    yield i

# t=test()
# for epoch in range(5):
#   t.next()
f=test1()
for epoch in range(3):
  for i in test1():
    print(i)
  print("***********")