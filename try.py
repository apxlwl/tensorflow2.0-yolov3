class father:
  def __init__(self):
    self.a = 1

  def change(self, s):
    self.a = 3


class son(father):
  def __init__(self):
    super().__init__()

  def change(self):
    self.a = 5
