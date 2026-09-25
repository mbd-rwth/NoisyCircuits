from contextlib import contextmanager
from contextvars import ContextVar
from EagleDecomposition import EagleDecomposition
from HeronDecomposition import HeronDecomposition


DEFAULT_GATE_SET = {
    "eagle" : {
        "basis_gates" : [["x", "sx", "rz"], ["ecr"]],
        "gate_decomposition" : EagleDecomposition
    },
    "heron" : {
        "basis_gates" : [["x", "sx", "rz", "rx"], ["rzz", "cz"]],
        "gate_decomposition" : HeronDecomposition
    }
}

_basis_gate_set : ContextVar[dict] = ContextVar(
    "basis_gate_set",
    default = DEFAULT_GATE_SET
)

def get_map() -> dict:
    return _basis_gate_set.get()

@contextmanager
def use_basis_gate_set(
    custom_set : dict,
    merge : bool = True
):
    new_gate_set = {**DEFAULT_GATE_SET, **custom_set} if merge else custom_set
    token = _basis_gate_set.set(new_gate_set)
    try:
        yield
    finally:
        _basis_gate_set.reset(token)