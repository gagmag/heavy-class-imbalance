import numpy as np
import math
import torch


def get_imbalanced_data(config, data_type="same inner"):

    c = config.c
    n = config.n
    d = config.d
    alpha = config.alpha
    seed = config.seed
    group_size = config.group_size

    np.random.seed(seed)



    #y = [0]*c + [(i+1) for i in range(c)] # class belongings of the datapoints



    y = []
    curr_class_num = 0

    for i in range(group_size):
        num_class = 2**i

        num_examples = 2**(group_size-i)

        for j in range(num_class):

            y += [curr_class_num]*num_examples

            curr_class_num += 1



    X = np.random.rand(n, d) # random unfirom in d dimension


    if data_type == "same inner":

        def generate_orthogonal_vectors(n, d):

            A = np.random.randn(d, n)

            Q, R = np.linalg.qr(A)

            return (Q[:, :n]).T


        A = generate_orthogonal_vectors(n+1,d)

        for i in range(n+1):
            A[i] = A[i] / np.linalg.norm(A[i])


        X = np.zeros((n,d))

        t = math.sqrt(alpha / (1-alpha))

        for i in range(n):
           
           X[i] = (A[i] + t*A[n]) / math.sqrt(1 + t**2)


    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)


    return X,y




""""
def get_general_data(alpha=0.0, seed = 100):

  np.random.seed(seed)


  y = [i for i in range(c+1)] # class belongings of the datapoints

  def generate_orthogonal_vectors(n, d):

      A = np.random.randn(d, n)

      Q, R = np.linalg.qr(A)

      return (Q[:, :n]).T


  A = generate_orthogonal_vectors(c+2,d)

  for i in range(c+1):
      A[i] = A[i] / np.linalg.norm(A[i])


  X = np.zeros((c+1,d))

  t = math.sqrt(alpha / (1-alpha))

  for i in range(c+1):

    X[i] = (A[i] + t*A[c+1]) / math.sqrt(1 + t**2)

    a = random.randint(0,1)
    if a ==1:
      X[i] = -X[i]


  X = torch.tensor(X, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.float32)

  return X,y

def get_general_data_array(alpha_array):
  
    X,y = get_general_data(alpha = alpha_array[0], seed=0)
  
    for i in range(1,len(alpha_array)):
  
      X_temp, y_temp = get_general_data(alpha = alpha_array[i], seed=i)
  
      X = torch.cat((X, X_temp), 0)
      y = torch.cat((y, y_temp), 0)
  
    return X,y

def get_bad_data(seed = 100):


  np.random.seed(seed)
  X = np.random.randn(c+1, d)
  y = [0 for i in range(c+1)]

  X = torch.tensor(X, dtype=torch.float32)
  y = torch.tensor(y, dtype=torch.float32)

  return X,y


"""