import numpy as np
import random


def get_matrix_dim_from_n_lower_tri(n_lower_tri: int)-> int:
    n = (np.sqrt(8 * n_lower_tri + 1) - 1) / 2
    if n.is_integer():
        return int(n)
    else:
        raise ValueError(f"The input {n_lower_tri} does not correspond to a valid square matrix dimension.")

def get_n_lower_tri_from_matrix_dim(n: int)-> int:
    return int(n*(n+1)/2)

def create_square_symmetric_matrix_from_lower_tri_array(a):
    n = get_matrix_dim_from_n_lower_tri(len(a))
    A = np.zeros_like(a, shape=(n,n))
    A[np.tril_indices(n)] = a
    A = A + A.T - np.diag(A.diagonal())
    return A


# def matrix_inverse(A):
#     n = len(A)
#     A = A.tolist()
#     I = np.identity(n).tolist()
    
#     for fd in range(n):
#         fdScaler = 1.0 / A[fd][fd]
#         for j in range(n):
#             A[fd][j] = A[fd][j] * fdScaler
#             I[fd][j] = I[fd][j] * fdScaler
#         for i in list(range(n))[0:fd] + list(range(n))[fd+1:]:
#             crScaler = A[i][fd]
#             for j in range(n):
#                 A[i][j] = A[i][j] - crScaler * A[fd][j]
#                 I[i][j] = I[i][j] - crScaler * I[fd][j]
#     return np.array(I)

# Reference https://stackoverflow.com/questions/32114054/matrix-inversion-without-numpy
def transposeMatrix(m):
    return list(map(list,zip(*m)))

def getMatrixMinor(m,i,j):
    return [row[:j] + row[j+1:] for row in (m[:i]+m[i+1:])]

def getMatrixDeternminant(m):
    #base case for 2x2 matrix
    if len(m) == 2:
        return m[0][0]*m[1][1]-m[0][1]*m[1][0]

    determinant = 0
    for c in range(len(m)):
        determinant += ((-1)**c)*m[0][c]*getMatrixDeternminant(getMatrixMinor(m,0,c))
    return determinant

def matrix_inverse(m):
    if isinstance(m, np.ndarray):
        m = m.tolist()
    determinant = getMatrixDeternminant(m)
    #special case for 2x2 matrix:
    if len(m) == 2:
        return [[m[1][1]/determinant, -1*m[0][1]/determinant],
                [-1*m[1][0]/determinant, m[0][0]/determinant]]

    #find matrix of cofactors
    cofactors = []
    for r in range(len(m)):
        cofactorRow = []
        for c in range(len(m)):
            minor = getMatrixMinor(m,r,c)
            cofactorRow.append(((-1)**(r+c)) * getMatrixDeternminant(minor))
        cofactors.append(cofactorRow)
    cofactors = transposeMatrix(cofactors)
    for r in range(len(cofactors)):
        for c in range(len(cofactors)):
            cofactors[r][c] = cofactors[r][c]/determinant
    return np.array(cofactors)