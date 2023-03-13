from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TransformerID:
    """
    This is an identifier that points to a Transformer in a Pipeline.

    Attributes
    ----------
    t_index : int
        Unique identifier value
    """

    t_index: int
    name: str

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.name}#{self.t_index}"

    def __hash__(self) -> int:
        return hash(str(self))

    @staticmethod
    def from_str(t_id: str) -> TransformerID:
        name, index = t_id.split("#")
        return TransformerID(t_index=int(index), name=name)
