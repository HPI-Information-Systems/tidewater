from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Dict, Type, Union, List, TypeVar, Sequence

from pydantic import create_model

from tidewater.datatypes.base import BaseDataType, TidewaterTypeConfig
from tidewater.datatypes import TimeSeries

InterfaceID = str
InterfaceValue = Union[BaseDataType, Sequence[BaseDataType]]
InterfaceValueType = TypeVar("InterfaceValueType", bound=InterfaceValue)


class Interface(ABC):
    """
    The base class for every Transformer Interface.
    Interfaces are necessary for the communication of Transformers
    in a Pipeline. An Interface can either be an input
    or an output Interface.

    Attributes
    ----------
    _attr : Dict[InterfaceID, Type[InterfaceValue]]
        An object for defining interface values
    _values : Dict[InterfaceID, Optional[InterfaceValue]]
        An object for storing interface values
    """

    def __init__(self, **attr: Type[InterfaceValueType]) -> None:
        """
        The initialization of an Interface only defines the data types.

        In the below example, an Interface subclass _SomeInterface_ has one attribute named *key_name*
        that can be filled with values of type TimeSeries.

        Example
        -------
        >>> SomeInterface(key_name=TimeSeries)

        Parameters
        ----------
        attr : Dict[InterfaceID, Type[InterfaceValue]]
            An object for defining interface values
        """

        self._model_def = create_model(
            __model_name="InterfaceDefinitions",
            __config__=TidewaterTypeConfig,
            **{k: (Optional[v], None) for k, v in attr.items()},
        )  # type: ignore  # does not get the correct parameter definitions
        self._model = self._model_def()

    def get_value(self, key: InterfaceID) -> Optional[InterfaceValue]:
        """
        Gets the interface value for the given key.

        Parameters
        ----------
        key : InterfaceID
            The key of the value to be returned

        Returns
        -------
        Optional[InterfaceValue]
            Eventually set value for the given key
        """

        attr: Optional[InterfaceValue] = getattr(self._model, key)
        return attr

    def set_value(self, key: InterfaceID, value: Optional[InterfaceValue]) -> None:
        """
        Sets the interface value for the given key and checks whether the data type is valid based on the definitions.

        Parameters
        ----------
        key : InterfaceID
            The key of the value to be set
        value : Optional[InterfaceValue]
            The value to be set
        """

        setattr(self._model, key, value)

    def set_values(self, values: Dict[InterfaceID, Optional[InterfaceValue]]) -> None:
        """
        Sets multiple interface values for the given keys and checks
        whether the data types are valid based on the definitions.

        Parameters
        ----------
        values : Dict[InterfaceID, Optional[InterfaceValue]]
            The values to be set
        """

        for k, v in values.items():
            self.set_value(k, v)

    @property
    @abstractmethod
    def is_input(self) -> bool:
        """
        Tells whether the current Interface is an input Interface.

        Returns
        -------
        bool
            Is the current Interface an input?
        """

        ...

    @property
    def is_output(self) -> bool:
        """
        Tells whether the current Interface is an output Interface.

        Returns
        -------
        bool
            Is the current Interface an output?
        """

        return not self.is_input

    def __repr__(self) -> str:
        return str(self)

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self._model})"

    def __contains__(self, key: InterfaceID) -> bool:
        return key in self._model

    def merge(self, other: Interface) -> Interface:
        """
        Generates a new Interface that is a merge of the current Interface and the given Interface.

        Parameters
        ----------
        other : Interface
            The Interface to be merged

        Returns
        -------
        Interface
            The merged Interface
        """

        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}")

        new_attr = {
            **{k: v.outer_type_ for k, v in self._model_def.__fields__.items()},
            **{k: v.outer_type_ for k, v in other._model_def.__fields__.items()},
        }
        new_interface = self.__class__(**new_attr)
        new_interface.set_values({**self._model.dict(), **other._model.dict()})
        return new_interface


class InputInterface(Interface):
    @property
    def is_input(self) -> bool:
        return True

    @property
    def all_available(self) -> bool:
        """
        Tells whether all values of the InputInterface are set.

        Returns
        -------
        bool
            Are all values set?
        """

        return all([v is not None for v in self._model.dict().values()])


class OutputInterface(Interface):
    @property
    def is_input(self) -> bool:
        return False
