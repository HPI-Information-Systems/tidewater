from __future__ import annotations

from dataclasses import dataclass, field
import redis
from typing import Optional, List, Iterator, Dict, Type, Any, Union
from .transformer_id import TransformerID
from ..datatypes import *
from ..datatypes.base import Serializer
import time
import joblib
from io import BytesIO
import numpy as np
import json
import docker
from docker.models.containers import Container
import uuid


JOBQUEUE = "job_queue"
REGISTERED_WORKERS = "registered_workers"


@dataclass
class Distributed:
    host: str = "localhost"
    port: int = 6379
    connection: Optional[redis.Redis] = None
    container: Optional[Container] = None
    scheduler: bool = False
    connection_trials: int = 5
    register_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def start_redis(self) -> None:
        client = docker.from_env()
        print("Docker pull Redis...")
        client.images.pull("redis", "7.0.5-bullseye")
        print("Docker start Redis...")
        self.container = client.containers.run(
            "redis", name="tidewater-redis", ports={"6379/tcp": str(self.port)}, remove=True, detach=True
        )

    def stop_redis(self) -> None:
        if self.container is not None:
            print("Docker stop Redis")
            self.container.stop()

    def connect(self) -> Distributed:
        pool = redis.ConnectionPool(host=self.host, port=self.port, db=0)
        self.connection = redis.Redis(connection_pool=pool)
        for _ in range(self.connection_trials):
            try:
                self.connection.ping()
                assert self.scheduler or self.count_jobs_left() > 0, "Job Queue should be filled."
            except redis.ConnectionError:
                print("Connection to Redis failed. Retry in 3 sec...")
                time.sleep(3)
                continue
            except AssertionError:
                print("Waiting for filled Job Queue...")
                time.sleep(3)
                continue
            else:
                if not self.scheduler:
                    self.register()
                return self
        raise ConnectionError(f"Tidewater could not connect to Redis at {self.host}:{self.port}!")

    def register(self) -> None:
        if self._is_connected():
            self.connection.hset(REGISTERED_WORKERS, str(self.register_id), 0)  # type: ignore

    def finish(self) -> None:
        if self._is_connected():
            self.connection.hset(REGISTERED_WORKERS, str(self.register_id), 1)  # type: ignore

    def wait_for_redis_shutdown(self) -> None:
        if self._is_connected():
            try:
                while self.connection.ping():  # type: ignore
                    time.sleep(1)
            except redis.exceptions.ConnectionError:
                pass

    def cluster_finished(self) -> bool:
        if self._is_connected():
            return all([self.connection.hget(REGISTERED_WORKERS, hkey).decode() == "1" for hkey in self.connection.hkeys(REGISTERED_WORKERS)])  # type: ignore
        return True

    def _is_connected(self, raises: bool = True) -> bool:
        if self.connection is None:
            err = "The distributed Pipeline is not connected to Redis."
            if raises:
                raise ConnectionError(err)
            print(err)

        return True

    def add_execution_order(self, execution_order: List[TransformerID]) -> None:
        if self.scheduler and self._is_connected():
            self.connection.rpush(JOBQUEUE, *list(map(str, execution_order)))  # type: ignore

    def get_next_transformer(self) -> Optional[TransformerID]:
        if self._is_connected():
            t_id = self.connection.lpop(JOBQUEUE)  # type: ignore
            if t_id is not None:
                return TransformerID.from_str(t_id.decode())
        return None

    def count_jobs_left(self) -> int:
        if self._is_connected():
            return self.connection.llen(JOBQUEUE)  # type: ignore
        return 0

    def iter_jobs(self) -> Iterator[TransformerID]:
        try:
            t_id = self.get_next_transformer()
            while t_id is not None:
                yield t_id
                t_id = self.get_next_transformer()
        except redis.exceptions.ConnectionError:
            print("The connection to Redis does not exist anymore.")

    def _get_key_names(self, t_id: TransformerID) -> str:
        return f"input_values:{t_id}"

    def set_input_value(
        self, t_id: TransformerID, interface_name: str, interface_value: Union[List[BaseDataType], BaseDataType]
    ) -> None:
        if self._is_connected():
            self.connection.hset(self._get_key_names(t_id), interface_name, Serializer.serialize(interface_value))  # type: ignore

    def set_input_values(
        self, t_id: TransformerID, **interface_values: Union[List[BaseDataType], BaseDataType]
    ) -> None:
        for k, v in interface_values.items():
            self.set_input_value(t_id, k, v)

    def get_input_values(self, t_id: TransformerID) -> Dict[str, Union[List[BaseDataType], BaseDataType]]:
        if self._is_connected():
            t_name = self._get_key_names(t_id)
            keys = self.connection.hkeys(t_name)  # type: ignore
            return {key.decode(): Serializer.deserialize(self.connection.hget(t_name, key)) for key in keys}  # type: ignore
        return dict()

    def __enter__(self) -> None:
        if self.scheduler:
            self.start_redis()
        self.connect()

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        if not self.scheduler and self._is_connected():
            self.finish()
            self.wait_for_redis_shutdown()
        if self.scheduler:
            while not self.cluster_finished():
                time.sleep(3)
            self.connection.close()  # type: ignore
            self.stop_redis()
