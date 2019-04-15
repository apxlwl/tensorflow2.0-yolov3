class father:
  def __init__(self,name):
    self.name=3

class son(father):
  def __init__(self,name,score):
    super().__init__(name)
    self.score=score

s=son(2,3)
print(s.name)
print(s.score)