# Tidewater

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python version 3.7|3.8|3.9](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)]()



A Pipeline for the Holistic Analysis of Time Series Anomalies

## Etymology

The __Tidewater__ pipeline was the world's first long-distance pipeline. It was used to transport oil, because transporting it via train was too expensive.

## Installation

Clone repository and install into your active Python environment.

```shell
make install
```

## Tests

```shell
make test
```

## Paper Experiments

[Download](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/) the UCR Archive into `~/datasets/UCRArchive_2018` or change it in the [experiments/ucr_experiments.sh](./experiments/ucr_experiments.sh) file.

To start the experiments on the UCR Archive, run the following script:

```shell
make experiments
```

## References

> Dau, H. A., Keogh, E., Kamgar, K., Yeh, C.-C. M., Zhu, Y., Gharghabi, S., Ratanamahatana, C. A., Yanping, Hu, B., Begum, N., Bagnall, A., Mueen, A., Batista, Gustavo, & Hexagon-ML. (2018). The UCR Time Series Classification Archive.