from ndlists import ndlist
from typing import List

from math import cos, sin, sqrt, pi

"""
Replace with your name and the one of your working partner, otherwise you won't be evaluated
Respect the syntax "firstname_lastname"
"""
student_name = ["pablo_gonzalez_lazaro", "alberto_lopez_del_amo"]

# name for file: TME1_functions_6.py


# Paste here all the function you implemented in the TME1 notebook
# Exercise 1
def _ket(lst: list) -> 'ndlist':
    """
    Create a ket from a list.
    :param lst: list of elements
    :return: ndlist representing the ket
    """
    ket = [[i] for i in lst]
    return ndlist(ket)



def _norm(array: ndlist) -> float:
    """
    Compute the norm of a ndlist object.
    :param array: ndlist object
    :return: norm (float)
    """
    norm_squared = 0
    for i in array:
        #Sum the squared module of the components
        norm_squared += sqrt((i[0].real)**2 + (i[0].imag)**2)**2
    #Square root of the sum
    norm = sqrt(norm_squared)
    return norm


def _scalar_mult(scalar: complex, array: ndlist) -> 'ndlist':
    """
    Multiply a ndlist object by a scalar.
    :param scalar: scalar
    :param array: ndlist
    :return: ndlist object after multiplication
    """
    result = []
    for i in range(array.shape[0]):
        if (array.ndim > 1):
            result.append(_scalar_mult(scalar, ndlist(array[i])))
        else:
            result.append(scalar * array[i])

    return ndlist(result)


def _normalize(array: ndlist) -> 'ndlist':
    """
    Normalize a ndlist object.
    :param array: ndlist object
    :return: normalized ndlist object
    """
    result = []
    norm = _norm(array)
    for i in range(array.shape[0]):
        result.append([array[i][0]/norm])
    return ndlist(result)


# Exercise 2
def _zeros(n: int, m: int) -> 'ndlist':
    """
    Create a zero matrix of shape (n, m).
    :param n: number of rows
    :param m: number of columns
    :return: ndlist representing the zero matrix
    """
    zero_list = []
    for i in range(n):
        zero_list.append([])
        for j in range(m):
            zero_list[i].append(0)
    
    return ndlist(zero_list)


def _identity(n: int) -> 'ndlist':
    """
    Create an identity matrix of shape (n, n).
    :param n: size of the identity matrix
    :return: ndlist representing the identity matrix
    """
    identity_list = []
    for i in range(n):
        identity_list.append([])
        for j in range(n):
            if i == j:
                identity_list[i].append(1)
            else:
                identity_list[i].append(0)
    return ndlist(identity_list)


def _matrix(lvec: List[ndlist]) -> 'ndlist':
    """
    Create a matrix from a list of lists using the zero matrix function.
    :param lvec: list of lists representing the matrix
    :return: ndlist representing the matrix
    """
    return_matrix = _zeros(len(lvec), len(lvec[0]))
    for i in range(return_matrix.shape[0]):
        for j in range(return_matrix.shape[1]):
            return_matrix[i][j] = lvec[i][j][0]
    return return_matrix


def _matmul(A: ndlist, B: ndlist) -> 'ndlist':
    """
    Perform matrix multiplication A @ B.
    :param A: ndlist representing matrix A
    :param B: ndlist representing matrix B
    :return: ndlist representing the result of A @ B
    """
    if A.shape[1] != B.shape[0]:
        #Matrix multiplication not possible
        return
    
    result_list = _zeros(A.shape[0], B.shape[1])
    for i in range(A.shape[0]):
        #1 iteration -> 1 row of A
        for j in range(B.shape[1]):
            #1 iteration -> 1 column of B
            value = 0
            for k in range(B.shape[0]):
                #1 iteration -> 1 element of column of B
                value += A[i,k] * B[k,j]
            result_list[i,j] = value
    return ndlist(result_list)


def _det(matrix: ndlist) -> float:
    """
    Compute the determinant of a square matrix.
    :param matrix: ndlist representing a square matrix
    :return: determinant (float)
    """
    #Laplace's expansion based solution
    if matrix.shape[0] == 1:
        if matrix.ndim == 1:
            return matrix[0]
        if matrix.shape[1] == 1:
            return matrix[0][0]
        else:
            return None
    
    determinant = 0
    for i in range(matrix.shape[0]):
        minor = [row[1:] for row in matrix]
        del minor[i] 
        determinant += matrix[i][0] * (-1)**(i+0) * _det(ndlist(minor))
    return determinant


def _transpose(matrix: ndlist) -> 'ndlist':
    """
    Compute the transpose of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the transposed matrix
    """
    transpose = _zeros(matrix.shape[1], matrix.shape[0])
    for i in range(transpose.shape[0]):
        for j in range(transpose.shape[1]):
            transpose[i][j] = matrix[j][i]
    return transpose


def _hermitian(matrix: ndlist) -> 'ndlist':
    """
    Compute the Hermitian conjugate (conjugate transpose) of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the Hermitian conjugate
    """
    transpose = _transpose(matrix=matrix)
    for i in range(transpose.shape[0]):
        for j in range(transpose.shape[1]):
            if type(transpose[i][j]) == complex:
                transpose[i][j] = transpose[i][j].conjugate()
    return transpose


# Exercise 3
def is_unitary(matrix: ndlist, tol: float = 1e-9) -> bool:
    """
    Check if a matrix is unitary.
    :param matrix: ndlist object representing a matrix
    :param tol: tolerance level for floating point comparison
    :return: True if unitary, False otherwise
    """
    hermitian = _hermitian(matrix)
    mult = _matmul(hermitian, matrix)
    for i in range(mult.shape[0]):
        if (abs(mult[i][i] - 1) >= tol):
            return False
    return True


def apply_unitary(ket: ndlist, U: ndlist) -> 'ndlist':
    """
    Apply a unitary matrix U to a ket.
    :param ket: ndlist representing the ket
    :param U: ndlist representing the unitary matrix
    :return: ndlist representing the new ket
    """
    new_ket = []
    if U.shape[1] != ket.shape[0]:
        return
    else:
        for i in range(U.shape[0]):
            component = 0
            for j in range(U.shape[1]):
                component += U[i][j] * ket[j][0]
            new_ket.append(component)
    return _ket(new_ket)


# Exercise 4
def bra(ket: ndlist) -> 'ndlist':
    """
    Compute the bra associated to a ket.
    :param ket: ndlist representing the ket
    :return: ndlist representing the bra
    """
    bra = []
    for i in ket:
        bra.append(i[0].real - 1j*i[0].imag)

    return ndlist([bra])


# Exercise 5
def _inner(ket1: ndlist, ket2: ndlist) -> complex:
    """
    Compute the inner product between two kets.
    :param ket1: ndlist representing the first ket
    :param ket2: ndlist representing the second ket
    :return: inner product (complex)
    """
    bra1 = bra(ket1)
    inner = 0
    for i in range(bra1.shape[1]):
        inner += bra1[0][i] * ket2[i][0]
    return inner


# Exercise 6
def measure_in_basis(ket: ndlist, basis: ndlist) -> List[float]:
    """
    Compute the probabilities of measuring the ket in the given basis.
    :param ket: ndlist representing the ket
    :param basis: ndlist representing the basis as a matrix
    :return: list of probabilities
    """
    
    probabilities = []
    for i in basis:
        inner_product = _inner(_ket(i), ket)
        probabilities.append(inner_product * inner_product.conjugate())
    return probabilities


