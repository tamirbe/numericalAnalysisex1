"""
Nir Agiv 208150607
Tamir Ben Eden 322384603
"""
import numpy as np


def is_invertible(matrix):
    # Check if the matrix is already a NumPy array
    if not isinstance(matrix, np.ndarray):
        # Convert the matrix to a NumPy array if it isn't already
        matrix = np.array(matrix)

    # Ensure the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be square (nxn)")

    # Calculate the determinant
    determinant = np.linalg.det(matrix)

    # Check if the determinant is non-zero
    return determinant != 0


def invert_matrix(matrix):
    # Ensure the matrix is square
    n = matrix.shape[0]
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("The matrix must be square (nxn)")

    # Create an identity matrix of the same size
    identity_matrix = np.eye(n)

    # Augment the matrix with the identity matrix
    augmented_matrix = np.hstack((matrix, identity_matrix))

    # Perform Gaussian elimination
    for i in range(n):
        # Ensure the pivot element is non-zero by swapping rows if necessary
        if augmented_matrix[i, i] == 0:
            for j in range(i+1, n):
                if augmented_matrix[j, i] != 0:
                    augmented_matrix[[i, j]] = augmented_matrix[[j, i]]
                    break
            else:
                raise ValueError("Matrix is singular and cannot be inverted.")

        # Normalize the pivot row
        augmented_matrix[i] = augmented_matrix[i] / augmented_matrix[i, i]

        # Eliminate the current column in other rows
        for j in range(n):
            if i != j:
                augmented_matrix[j] = augmented_matrix[j] - augmented_matrix[i] * augmented_matrix[j, i]

    # The right half of the augmented matrix is now the inverse
    inverse_matrix = augmented_matrix[:, n:]

    return inverse_matrix


def max_norm(matrix):
    # Initialize max_row_sum to 0
    max_row_sum = 0

    # Iterate over each row
    for row in matrix:
        # Calculate the sum of absolute values in the current row
        row_sum = sum(abs(element) for element in row)
        # Update max_row_sum if the current row_sum is greater
        max_row_sum = max(max_row_sum, row_sum)

    return max_row_sum


def condition_number(matrix):
    # Step 1: Calculate the max norm (infinity norm) of A
    matrix_norm = max_norm(matrix)

    # Step 2: Calculate the inverse of A
    matrix_invert = invert_matrix(matrix)

    # Step 3: Calculate the max norm of the inverse of A
    matrix_norm_invert = max_norm(matrix_invert)

    # Step 4: Compute the condition number
    cond = matrix_norm * matrix_norm_invert

    return cond


if __name__ == '__main__':
    A = np.array([[1, -1, -2],
                  [2, -3, -5],
                  [-1, 3, 5]])

    if not is_invertible(A):
        raise ValueError("Matrix isn't invertible")

    cond = condition_number(A)
    print(cond)

