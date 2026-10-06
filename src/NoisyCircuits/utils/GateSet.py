from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from .EagleDecomposition import EagleDecomposition
from .HeronDecomposition import HeronDecomposition


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

class _BasisGateSetProxy:
    def __getitem__(
            self, 
            key : str
        ) -> dict:
        return get_map()[key]

    def __contains__(
            self,
            key : str
        ) -> str:
        return key in get_map()

    def keys(self) -> list[str]:
        return get_map().keys()

    def values(self)->Iterable:
        return get_map().values()

    def items(self)->Iterable:
        return get_map().items()

    def __iter__(self) -> Iterator:
        return iter(get_map())

    def __len__(self) -> int:
        return len(get_map())

    def __repr__(self) -> str:
        return repr(get_map())

basis_gate_set = _BasisGateSetProxy()