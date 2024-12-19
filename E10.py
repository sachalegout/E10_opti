import numpy as np
from scipy.optimize import linprog

class BranchAndBoundSolver:
    def __init__(self, revenues, days, total_days):
        self.revenues = revenues
        self.days = days
        self.total_days = total_days
        self.best_solution = None
        self.best_value = -np.inf

    def solve(self):
        num_projects = len(self.revenues)
        bounds = [(0, 1) for _ in range(num_projects)]
        solution, value = self._solve_lp_relaxation(bounds)
        if solution is not None:
            self._branch_and_bound(solution, value, bounds)
        return self.best_solution, self.best_value

    def _solve_lp_relaxation(self, bounds):
        c = -np.array(self.revenues)
        A = [self.days]
        b = [self.total_days]
        result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
        if result.success:
            return result.x, -result.fun
        return None, None

    def _branch_and_bound(self, solution, value, bounds):
        if all(x.is_integer() for x in solution):
            if value > self.best_value:
                self.best_solution = solution
                self.best_value = value
            return
        fractional_index = next(i for i, x in enumerate(solution) if not x.is_integer())
        left_bounds = bounds.copy()
        left_bounds[fractional_index] = (0, np.floor(solution[fractional_index]))
        right_bounds = bounds.copy()
        right_bounds[fractional_index] = (np.ceil(solution[fractional_index]), 1)
        left_solution, left_value = self._solve_lp_relaxation(left_bounds)
        if left_solution is not None and left_value > self.best_value:
            self._branch_and_bound(left_solution, left_value, left_bounds)
        right_solution, right_value = self._solve_lp_relaxation(right_bounds)
        if right_solution is not None and right_value > self.best_value:
            self._branch_and_bound(right_solution, right_value, right_bounds)

revenues = [15, 20, 5, 25, 22, 17]
days = [51, 60, 35, 60, 53, 10]
total_days = 365

solver = BranchAndBoundSolver(revenues, days, total_days)
best_solution, best_value = solver.solve()

print("Best solution:", best_solution)
print("Best value:", best_value)
