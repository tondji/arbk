import numpy as np
import random

#@title : Implementation of differents methods: BK, ARBK and NRBK


def Kaczmarz_methods(A, b, y_start, x_true, max_iter, lbda, nber_block, alpha, funct_values, errors, residuals, sparsity_sol, tol, method, p_list=[]):

  # The Block Accelerated Randomized Bregman Kaczmarz method
  # for academic purpose; uses x_true to calculate the exact beta needed for the adaptive stepsize
  # A: The given matrix.
  # y_start: The starting point of the algorithm.
  # b: The true right hand size.
  # max_iter: The maximum number of iterations.
  # lbda: The sparsity parameter.
  # nber_block: The number of blocks the user want to use.
  # alpha \in [0,1]: parameter to use uniform probabilities or probabilities over the block row norm squared. (see Eq 3.7)
  # tol: The tolerance the user want to have.
  # p_list: The partition of the matrix A.

  m, n = A.shape
  theta = 1/nber_block

  # Initialize variables
  z_k = y_start
  y_k = y_start

  # for Nesterov Acceleration
  a = 1/nber_block
  B = 2
  v_k = y_start

  k = len(errors)

  # Constrution of the U_i matrices and the probabilities vectors
  if len(p_list) == 0:
    print('Creating the partition')
    index_list = list([i for i in range(m)])
    copy_index_list = index_list.copy()
    np.random.shuffle(copy_index_list)
    p_list = np.array_split(copy_index_list, nber_block)
  I = np.identity(m)
  squared_block_row_norms = []
  for blc in range(nber_block):
    idx = p_list[blc]
    idx.sort()
    squared_block_row_norms.append(np.linalg.norm(A[idx], ord=2)**2)
  probabilities = [(norm**alpha) for norm in squared_block_row_norms]
  probabilities = [(norm**alpha)/sum(probabilities) for norm in probabilities]

  for iter in range(max_iter):

    if iter ==0:
      print(f'Method = {method} : number of blocks = {nber_block}, alpha = {alpha}, lambda = {lbda} for {max_iter} iterations')

    # sample the block row index
    tauk = random.choices([j for j in range(nber_block)], weights = probabilities, k = 1)
    i = tauk[0]
    index = p_list[i]
    index.sort()
    U_i = I[index].T

    if method == 'ARBK':

      v_k = (1-theta)*y_k + theta*z_k
      update_arbk = (A[index] @ soft_skrinkage(A.T @ v_k, lbda) - b[index])/((squared_block_row_norms[i])*theta*nber_block)
      new_z_k = z_k - U_i @ update_arbk
      y_k = v_k + nber_block*theta*(new_z_k - z_k)

      theta = 0.5*(np.sqrt(theta**4 + 4*(theta**2)) - theta**2)
      z_k = new_z_k

      if iter % m ==0 :
        x_k = soft_skrinkage(A.T @ y_k, lbda)
        sparsity_sol.append(n - np.sum(x_k == np.zeros((n,1))))
        f_v_y = 0.5*(np.linalg.norm(x_k)**2) - b.T @ y_k
        funct_values.append(f_v_y[0][0])
        errors.append(np.linalg.norm(x_k  - x_true)/ (np.linalg.norm(x_true)))
        residuals.append(np.linalg.norm(A @ x_k  - b)/ (np.linalg.norm(b)))
        k +=1

      if min(errors[k-1],residuals[k-1]) < tol :
        # stopped = True
        break

    if method == 'BK':

      update_bk = (A[index] @ soft_skrinkage(A.T @ y_k, lbda) - b[index])/(squared_block_row_norms[i])
      y_k += - U_i @ update_bk

      if iter % m ==0 :
        x_k = soft_skrinkage(A.T @ y_k, lbda)
        sparsity_sol.append(n - np.sum(x_k == np.zeros((n,1))))
        f_v = 0.5*(np.linalg.norm(x_k)**2) - b.T @ y_k
        funct_values.append(f_v[0][0])
        errors.append(np.linalg.norm(x_k  - x_true)/ (np.linalg.norm(x_true)))
        residuals.append(np.linalg.norm(A @ x_k  - b)/ (np.linalg.norm(b)))
        k += 1

      if min(errors[k-1],residuals[k-1]) < tol :
          # stopped = True
          break

    if method == 'NRBK':

      Delta = 1/(nber_block**2) + 4*a/B  # Discriminant
      gamma = 0.5*(1/nber_block + np.sqrt(Delta))
      alpha = 1/(nber_block*gamma)
      beta = 1

      v_k = alpha*z_k + (1-alpha)*y_k
      update_nrbk = (A[index] @ soft_skrinkage(A.T @ v_k, lbda) - b[index])/(squared_block_row_norms[i])
      y_k = v_k - U_i @ update_nrbk
      z_k = beta*z_k + (1-beta)*v_k - gamma * U_i @ update_nrbk

      B = B/np.sqrt(beta)
      a = gamma * B
      if iter % m ==0 :
        x_k = soft_skrinkage(A.T @ y_k, lbda)
        sparsity_sol.append(n - np.sum(x_k == np.zeros((n,1))))
        f_v_y = 0.5*(np.linalg.norm(x_k)**2) - b.T @ y_k
        funct_values.append(f_v_y[0][0])
        errors.append(np.linalg.norm(x_k  - x_true)/ (np.linalg.norm(x_true)))
        residuals.append(np.linalg.norm(A @ x_k  - b)/ (np.linalg.norm(b)))
        k += 1

      if min(errors[k-1],residuals[k-1]) < tol :
          # stopped = True
          break

  return y_k, funct_values, errors, residuals, sparsity_sol

# Create a CT
def myphantom(N):
  # Adapted from:
  # Peter Toft, "The Radon Transform - Theory and Implementation", PhD
  # thesis, DTU Informatics, Technical University of Denmark, June 1996.
  # Translated from MATLAB to Python by ChatGPT
  xn = ((np.arange(N) - (N - 1) / 2) / ((N - 1) / 2)).reshape(-1, 1)
  Xn = np.tile(xn, (1, N))
  Yn = np.rot90(Xn)
  X = np.zeros((N, N))

  e = np.array([
    [1, 0.69, 0.92, 0, 0, 0],
    [-0.8, 0.6624, 0.8740, 0, -0.0184, 0],
    [-0.2, 0.1100, 0.3100, 0.22, 0, -18],
    [-0.2, 0.1600, 0.4100, -0.22, 0, 18],
    [0.1, 0.2100, 0.2500, 0, 0.35, 0],
    [0.1, 0.0460, 0.0460, 0, 0.1, 0],
    [0.1, 0.0460, 0.0460, 0, -0.1, 0],
    [0.1, 0.0460, 0.0230, -0.08, -0.605, 0],
    [0.1, 0.0230, 0.0230, 0, -0.606, 0],
    [0.1, 0.0230, 0.0460, 0.06, -0.605, 0]
  ])

  for i in range(e.shape[0]):
    a2 = e[i, 1] ** 2
    b2 = e[i, 2] ** 2
    x0 = e[i, 3]
    y0 = e[i, 4]
    phi = np.radians(e[i, 5])
    A = e[i, 0]
    x = Xn - x0
    y = Yn - y0
    index = np.where(
      ((x * np.cos(phi) + y * np.sin(phi)) ** 2) / a2
      + ((y * np.cos(phi) - x * np.sin(phi)) ** 2) / b2
      <= 1
    )
    X[index] += A

  X = X.ravel()
  X[X < 0] = 0
  return X

# The soft shrinkage function
def soft_skrinkage(x, lbda):
  return np.sign(x) * np.maximum(np.abs(x) - lbda, 0)