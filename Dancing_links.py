import numpy as np


def solve_sudoku(grid):
    def add_constraints(grid):
        N = 9
        constraints = []
        for row in range(N):
            for col in range(N):
                if grid[row][col] == 0:
                    for num in range(1, N + 1):
                        constraints.append((row, col, num))
        return constraints

    def exact_cover(X, Y, solution):
        if not X:
            yield list(solution)
        else:
            c = min(X, key=lambda c: len(X[c]))
            for r in list(X[c]):
                solution.append(Y[r])
                cols = [col for col in Y[r] if col in X]
                for col in cols:
                    X.remove(col)
                for j in Y[r]:
                    for i in X[j]:
                        X[j].remove(i)
                for s in exact_cover(X, Y, solution):
                    yield s
                solution.pop()
                for j in Y[r]:
                    for i in X[j]:
                        X[j].add(i)
                for col in cols:
                    X.add(col)

    def build_matrix(constraints):
        row_idx = 0
        X = set(constraints)
        Y = {}
        for r, c, num in constraints:
            row = [r, c, (r // 3 * 3) + c // 3, (r % 3 * 3) + c % 3, num - 1]
            for i in row:
                if i not in Y:
                    Y[i] = set()
                Y[i].add(row_idx)
            row_idx += 1
        return X, Y

    constraints = add_constraints(grid)
    X, Y = build_matrix(constraints)
    for solution in exact_cover(X, Y, []):
        for (row, col, num) in solution:
            grid[row][col] = num + 1
    return grid


# Example usage
print("got here")
grid = np.array([
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
])
solved_grid = solve_sudoku(grid)
print(solved_grid)
