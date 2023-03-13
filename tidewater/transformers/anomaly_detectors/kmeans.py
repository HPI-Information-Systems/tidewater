from .adapters import UnsupervisedDockerAnomalyDetector


class KMeans(UnsupervisedDockerAnomalyDetector):
    def __init__(self) -> None:
        super().__init__(image_name="registry.gitlab.hpi.de/akita/i/kmeans", group_privileges="phillip")
