from __future__ import annotations

from pathlib import Path
from os import PathLike
from typing import Union

from sklearn.base import TransformerMixin
import pickle

from tidewater.datatypes.base import BaseDataType


class Model(BaseDataType):
    """
    This is a wrapped ML model.

    Attributes
    ----------
    path : Union[Path, PathLike]
        Path where the model resides
    """

    path: Union[Path, PathLike]


class SKLearnModel(Model):
    def materialize(self) -> TransformerMixin:
        with open(self.path, "rb") as f:
            model = pickle.load(f)
        return model

    @staticmethod
    def from_transformer(skl_model: TransformerMixin, path: Union[Path, PathLike]) -> SKLearnModel:
        with open(path, "wb") as f:
            pickle.dump(skl_model, f)
        return SKLearnModel(path=path)
