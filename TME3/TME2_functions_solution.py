import ndlists as nd

# import TME1_functions as tme1  # Uncomment and change the module name if you have TME1 functions from your previous work or solutions. YOU NEED TO UNCOMMENT THIS LINE TO USE TME1 FUNCTIONS
import TME1_functions_solution as tme1  # Using the provided solutions for TME1 functions

from math import sqrt, pi, cos, sin, tan, exp
from typing import List

from random import random


"""
Replace with your name and the one of your working partner, otherwise you won't be evaluated
Respect the syntax "firstname_lastname". If you have several names it is simply "pierre_paul_jacques_dupont"
"""
student_name = ["name1", "name2"]  # Replace with your names

# Paste here all the functions you implemented in the TME2 notebook as well as the different examples you made to test them


# Exercise 1
def tensor_product(A: nd.ndlist, B: nd.ndlist) -> nd.ndlist:
    """
    Compute the tensor product of two ndlists recursively.
    :param A:
    :param B:
    :return: The tensor product A ⊗ B as a ndlist.
    """
    # Get the shapes of A and B
    n, m = A.shape
    p, q = B.shape

    result = tme1._zeros(n * p, m * q)

    for i in range(n):
        for j in range(m):
            # A[i, j] is a scalar, B is an ndlist
            # Multiply each element of B by A[i, j]
            for x in range(p):
                for y in range(q):
                    result[i * p + x, j * q + y] = A[i, j] * B[x, y]

    return result


# Exercise 2
def projector(state: nd.ndlist) -> nd.ndlist:
    """
    Construct the projector |ψ><ψ| for a normalized state vector ψ.
    """
    bra = tme1._bra(state)
    return tme1._matmul(state, bra)


def measurement_probability(state: nd.ndlist, projector: nd.ndlist) -> float:
    """
    Compute probability of obtaining outcome associated with projector
    when measuring state.
    """
    bra = tme1._bra(state)
    temp = tme1._matmul(projector, state)
    prob = tme1._matmul(bra, temp)
    return prob[0, 0].real  # Return the real part of the probability


# Simulation of repeated measurements
def simulate_measurements(state: nd.ndlist, projectors: List[nd.ndlist], n: int) -> List[float]:
    """
    Simulate n projective measurements on the given state using the provided projectors.
    Returns a list of measurement outcomes (indices of projectors).
    :param state: ndlist representing the quantum state (ket)
    :param projectors: list of ndlists representing the projectors
    :param n: number of measurements to simulate
    :return: list of measurement outcomes (indices)
    """
    outcomes = []
    probabilities = [measurement_probability(state, proj) for proj in projectors]

    for _ in range(n):
        r = random()
        cumulative_prob = 0.0
        for i, prob in enumerate(probabilities):
            cumulative_prob += prob
            if r < cumulative_prob:
                outcomes.append(i)
                break

    return outcomes