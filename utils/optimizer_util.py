import pickle
def save_opti(optimizer, pathfile):
  params_dict = {}
  config_dict = optimizer.get_config()
  weights_dict = {}
  symbolic_weights = optimizer.weights
  for var in symbolic_weights:
    weights_dict[var.name] = var.numpy()
  params_dict['config'] = config_dict
  params_dict['weights'] = weights_dict
  with open(pathfile, 'wb') as f:
    pickle.dump(params_dict, f)
  print("save optimizer success")

def load_opti(optimizer, pathfile):
  weightlist = []
  with open(pathfile, 'rb') as f:
    dict = pickle.load(f)
    optimizer.from_config(dict['config'])
    for k, v in dict['weights'].items():
      weightlist.append(v)
  optimizer.set_weights(weightlist)
  print("load optimizer success")