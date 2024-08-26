from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Set, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    left_vals = list(vals)
    right_vals = list(vals)
    left_vals[arg] -= epsilon / 2
    right_vals[arg] += epsilon / 2
    delta = f(*right_vals) - f(*left_vals)
    return delta / epsilon


variable_count = 1


# Variable is a Interface(All Scalar instance has implemented this interface)
class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    # DFS explanation and recursive implementation: https://youtu.be/PMMc4VsIacU?t=81
    marked: Set[int] = set()  # marked is visited node set(`Scalar` unique_id set)

    result: List[Variable] = []
    # visited: List[int] = list()

    def visit(v: Variable) -> None:
        result.append(v)

    def dfs(v: Variable) -> None:
        if (
            v.is_constant()
        ):  # if Scalar instance v has no ScalarHistory(history is None), it is constant
            return

        marked.add(v.unique_id)

        for w in v.parents:
            if w.unique_id not in marked:
                dfs(w)
        visit(v)

    dfs(variable)  # start from the right-most variable

    result = list(reversed(result))
    return result


# variable is Scalar instance, it has implemented Variable interface
def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # Output like(Computation Graph of test_backprop3):
    # ordered_vars = [Scalar(30.000000), Scalar(10.000000), Scalar(0.000000), Scalar(10.000000), Scalar(0.000000), Scalar(0.000000)]
    ordered_vars: Iterable[Variable] = topological_sort(variable)
    # Record the derivative of each variable
    # {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    derivatives: Dict[int, Any] = {var.unique_id: 0 for var in ordered_vars}
    # {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 5} (d_output of var4 is 5, unique_id of var4 is 6)
    derivatives[variable.unique_id] = deriv
    for var in ordered_vars:
        if var.is_leaf():
            var.accumulate_derivative(
                derivatives[var.unique_id]
            )  # Only leaf variables can have derivatives, so accumulate derivative for leaf variables in many iterations
        else:
            d_output = derivatives[var.unique_id]
            for parent_var, deriv in var.chain_rule(d_output):
                if parent_var.is_constant():
                    continue
                if parent_var.unique_id in derivatives:
                    derivatives[parent_var.unique_id] += deriv
                else:
                    derivatives[parent_var.unique_id] = deriv


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
