#!/usr/bin/env python3

A = [[3, 1, 4],
     [5, 2, -2]]
B = [[1, -1],
     [2, 0],
     [-6, 4]]

# Dot product using zip and reduce (sum) patterns.
def dot_product(A, B):
    return sum([x * y for x,y in zip(A, B)])

# Matrix multiplication using dot product.
def matrix_mult(A, B):
    if len(A) != len(B[0]):
        return None

    C = [[0 for x in range(len(A))] for x in range(len(A))]

    for r in range(len(C)):
        for c in range(len(C[0])):
            C[r][c] = dot_product(A[r], [x[c] for x in B])

    return C

print(A)
print(B)
print(matrix_mult(A, B))
