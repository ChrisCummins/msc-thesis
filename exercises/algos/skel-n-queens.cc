// Backtracking solution for N-queens problem. Adapted from "Geeks for
// Geeks".
//
// See: http://www.geeksforgeeks.org/backtracking-set-3-n-queen-problem/

#include <algorithm>
#include <iostream>

#define WIDTH  8
#define HEIGHT 8

#define MAX(x,y) ((x) > (y) ? (x) : (y))
#define MIN(x,y) ((x) < (y) ? (x) : (y))

#define NCOLS  MAX(WIDTH, HEIGHT)
#define NROWS  MIN(WIDTH, HEIGHT)

void print_board(bool board[NCOLS][NROWS]) {
    for (int i = 0; i < NCOLS; i++) {
        for (int j = 0; j < NROWS; j++) {
            if (board[i][j])
                std::cout << " Q ";
            else
                std::cout << " - ";
        }
        std::cout << "\n";
    }
}

// Check if a queen placed at board[row][col] can be attacked by any
// queens in the range board[0,row-1][0,col-1].
bool is_safe(bool board[NCOLS][NROWS], int row, int col) {
    int i, j;

    // Check this row on left side
    for (i = 0; i < col; i++)
        if (board[row][i])
            return false;

    // Check upper diagonal on left side
    for (i = row, j = col; i >= 0 && j >= 0; i--, j--)
        if (board[i][j])
            return false;

    // Check lower diagonal on left side
    for (i = row, j = col; j >= 0 && i < NCOLS; i++, j--)
        if (board[i][j])
            return false;

    return true;
}

// A recursive utility function to solve N Queen problem
bool n_queens(bool board[NCOLS][NROWS], int col) {
    // Base case: If all queens are placed then return true
    if (col >= NROWS)
        return true;

    // Recursion case: iteratively place queen in each square of
    // column
    for (int i = 0; i < NCOLS; i++) {
        // Check if queen can be placed on board[i][col]
        if (is_safe(board, i, col)) {
            // Place this queen in board[i][col]
            board[i][col] = true;

            // recur to place rest of the queens
            if (n_queens(board, col + 1))
                return true;

            // If placing queen in board[i][col] doesn't lead to a
            // solution then remove queen from board[i][col]
            board[i][col] = false; // BACKTRACK
        }
    }

     /* If queen can not be place in any row in this colum col
        then return false */
    return false;
}

// This function solves the N Queen problem using Backtracking.  It
// mainly uses n_queens() to solve the problem. It returns false if
// queens cannot be placed, otherwise return true and prints placement
// of queens in the form of 1s. Please note that there may be more
// than one solutions, this function prints one of the feasible
// solutions.
int main() {
    bool board[NCOLS][NROWS] = {{ false }};

    if (n_queens(board, 0)) {
        print_board(board);
        std::cout << "Solution found for " << NROWS << " queens.\n";
        return 0;
    }

    std::cout << "No solution found.\n";
    return 1;
}
