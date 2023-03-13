from __future__ import annotations
from io import BytesIO
import tempfile
import pandas as pd
from pydantic import BaseModel, BaseConfig
from pathlib import Path
from typing import Union, Any
from dataclasses import field
import joblib
import datetime as dt

import numpy as np


class TidewaterTypeConfig(BaseConfig):
    arbitrary_types_allowed = True
    validate_assignment = True


class Serializer:
    @staticmethod
    def serialize(obj: Any) -> bytes:
        buffer = BytesIO()
        joblib.dump(obj, buffer)
        buffer.seek(0)  # update to enable reading
        return buffer.read()

    @staticmethod
    def deserialize(b: bytes) -> Any:
        buffer = BytesIO(b)
        v = joblib.load(buffer)
        return v


class BaseDataType(BaseModel):
    """
    This is a base class for all datatypes.
    """

    class Config(TidewaterTypeConfig):
        pass


class RunTime(BaseDataType):
    """
    This is a RunTime data type holding the runtime of a previous transformer.
    """

    transformer_id: str
    run_time: dt.timedelta


class Scalar(BaseDataType):
    """
    This is a scalar data type that can be used to transfer results.

    Attributes
    ----------
    data : float
        Actual scalar value
    meta : dict
        Meta data
    """

    data: float
    meta: dict = {}

    def to_dataframe(self, **kwargs: Any) -> pd.DataFrame:
        meta = self.meta
        meta["scalar_value"] = self.data
        meta.update(kwargs)
        return pd.DataFrame([meta])


class NumpyType(BaseDataType):
    """
    This is a base class for all datatypes that are based on NumPy arrays.

    Attributes
    ----------
    data : np.ndarray
        Numpy data to be wrapped in NumpyType
    name : str
        Descriptive name of data
    """

    ndarray: np.ndarray
    name: str = ""

    def to_2d(self) -> np.ndarray:
        """
        Reshapes containing numpy data into a 2d shape.

        Returns
        -------
        np.ndarray
            Reshaped data
        """

        if self.ndarray.ndim == 1:
            return self.ndarray.reshape(-1, 1)
        elif self.ndarray.ndim == 2:
            return self.ndarray
        else:
            raise ValueError("Data should have at most 2 axes.")

    def flatten(self) -> np.ndarray:
        """
        Returns the numpy data as a flat array.

        Returns
        -------
        np.ndarray
            Flat data
        """

        return self.ndarray.flatten()

    def to_file(self, file: Path) -> None:
        """
        Saves numpy data to given file path.

        Parameters
        ----------
        file : Path
            Path where numpy data should be saved
        """

        np.savetxt(file, self.ndarray, delimiter=",")

    @classmethod
    def from_file(cls, file: Path) -> NumpyType:
        """
        Loads numpy data from given file path.

        Parameters
        ----------
        file : Path
            Path where numpy data resides

        Returns
        -------
        NumpyType
            Loaded and wrapped numpy data
        """

        return cls(ndarray=np.loadtxt(str(file), delimiter=","))

    def __str__(self) -> str:
        return f"{self.name}:\n{self.ndarray}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(data={repr(self.ndarray)}, name={repr(self.name)})"
