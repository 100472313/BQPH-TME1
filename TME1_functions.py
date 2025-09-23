from ndlists import ndlist
from typing import List

"""
Replace with your name and the one of your working partner, otherwise you won't be evaluated
Respect the syntax "firstname_lastname"
"""
student_name = ["firstname_lastname1", "firstname_lastname_2"]


# Paste here all the function you implemented in the TME1 notebook
# Exercise 1
def _ket(lst: list) -> 'ndlist':
    """
    Create a ket from a list.
    :param lst: list of elements
    :return: ndlist representing the ket
    """
    ...


def _norm(array: ndlist) -> float:
    """
    Compute the norm of a ndlist object.
    :param array: ndlist object
    :return: norm (float)
    """
    ...


def _scalar_mult(scalar: complex, array: ndlist) -> 'ndlist':
    """
    Multiply a ndlist object by a scalar.
    :param scalar: scalar
    :param array: ndlist
    :return: ndlist object after multiplication
    """
    ...


def _normalize(array: ndlist) -> 'ndlist':
    """
    Normalize a ndlist object.
    :param array: ndlist object
    :return: normalized ndlist object
    """
    ...


# Exercise 2
def _zeros(n: int, m: int) -> 'ndlist':
    """
    Create a zero matrix of shape (n, m).
    :param n: number of rows
    :param m: number of columns
    :return: ndlist representing the zero matrix
    """
    ...


def _identity(n: int) -> 'ndlist':
    """
    Create an identity matrix of shape (n, n).
    :param n: size of the identity matrix
    :return: ndlist representing the identity matrix
    """
    ...


def _matrix(lvec: List[ndlist]) -> 'ndlist':
    """
    Create a matrix from a list of lists using the zero matrix function.
    :param lvec: list of lists representing the matrix
    :return: ndlist representing the matrix
    """
    ...


def _matmul(A: ndlist, B: ndlist) -> 'ndlist':
    """
    Perform matrix multiplication A @ B.
    :param A: ndlist representing matrix A
    :param B: ndlist representing matrix B
    :return: ndlist representing the result of A @ B
    """
    ...


def _det(matrix: ndlist) -> float:
    """
    Compute the determinant of a square matrix.
    :param matrix: ndlist representing a square matrix
    :return: determinant (float)
    """
    ...


def _transpose(matrix: ndlist) -> 'ndlist':
    """
    Compute the transpose of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the transposed matrix
    """
    ...


def _hermitian(matrix: ndlist) -> 'ndlist':
    """
    Compute the Hermitian conjugate (conjugate transpose) of a matrix.
    :param matrix: ndlist representing a matrix
    :return: ndlist representing the Hermitian conjugate
    """
    ...


# Exercise 3
def is_unitary(matrix: ndlist, tol: float = 1e-9) -> bool:
    """
    Check if a matrix is unitary.
    :param matrix: ndlist object representing a matrix
    :param tol: tolerance level for floating point comparison
    :return: True if unitary, False otherwise
    """
    ...


def apply_unitary(ket: ndlist, U: ndlist) -> 'ndlist':
    """
    Apply a unitary matrix U to a ket.
    :param ket: ndlist representing the ket
    :param U: ndlist representing the unitary matrix
    :return: ndlist representing the new ket
    """
    ...


# Exercise 4
def bra(ket: ndlist) -> 'ndlist':
    """
    Compute the bra associated to a ket.
    :param ket: ndlist representing the ket
    :return: ndlist representing the bra
    """
    ...


# Exercise 5
def _inner(ket1: ndlist, ket2: ndlist) -> complex:
    """
    Compute the inner product between two kets.
    :param ket1: ndlist representing the first ket
    :param ket2: ndlist representing the second ket
    :return: inner product (complex)
    """
    ...


# Exercise 6
def measure_in_basis(ket: ndlist, basis: ndlist) -> List[float]:
    """
    Compute the probabilities of measuring the ket in the given basis.
    :param ket: ndlist representing the ket
    :param basis: ndlist representing the basis as a matrix
    :return: list of probabilities
    """
    ...


