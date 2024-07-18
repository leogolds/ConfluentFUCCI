# ConfluentFUCCI
A suite of tools for analyzing large scale confluent FUCCI experiments

ConfluentFUCCI has now been peer-reviewed and the publication is freely available [here](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0305491).  To cite please use: 

**Goldstien L, Lavi Y, Atia L (2024) ConfluentFUCCI for fully-automated analysis of cell-cycle progression in a highly dense collective of migrating cells. PLoS ONE 19(6): e0305491. https://doi.org/10.1371/journal.pone.0305491**

## Overview
This repo includes an integration and automation layer for running [CellPose](https://github.com/MouseLand/cellpose) (person-in-the-loop ML driven cell segentation) and TrackMate (cell tracking). Furthermore, a set of analysis and visualization tools for studying confluent cellular dynamics using a FUCCI stain are included.


## For Users
The recommended way for trying out ConfluentFUCCI is to use our prebuilt conainer image:

```shell
docker run -it --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8080:8080 \
    -p 9876:9876 \
    leogold/confluentfucci:latest
```

This will start a container that will serve ConfluentFUCCI on [localhost:8080](http://localhost:8080) and a virtual desktop on [localhost:9876](http://localhost:9876). The app served using the above command does not require a GPU, which significantly affects segmentation time. Too speed up segmentation by leveraging your [CUDA compatible GPU](https://developer.nvidia.com/cuda-gpus), please use:

```shell
docker run -it --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8080:8080 \
    -p 9876:9876 \
    --gpus all \
    leogold/confluentfucci:latest
```

### Using docker-compose
To simplify deployment, please check out our [docker-compose.yaml](https://github.com/leogolds/ConfluentFUCCI/blob/main/containers/confluentfucci/docker-compose.yaml). Placing this file in the same path as your data should allow you to test the app using:

```shell
docker compose up
```

If a [CUDA compatible GPU](https://developer.nvidia.com/cuda-gpus) is availble on your system, make sure to uncomment:

```shell
#    deploy:
#      resources:
#        reservations:
#          devices:
#            - driver: nvidia
#              count: 1
#              capabilities: [ gpu ]
```


## For Developers
This project is set up using poetry. To install the dependencies, run `poetry install` from the root of the project.

```shell
poetry install
```

To add a new dependency, run `poetry add <dependency>` from the root of the project.

```shell
poetry add <dependency>
```

### Testing
This project uses [pytest](https://docs.pytest.org/en/stable/) for testing. To run the tests, run `pytest` from the root of the project in the poetry shell.

```shell
poetry run pytest
```

There are sensible defaults for pytest setup in the `pyproject.toml` file. You can override these defaults by passing in command line arguments. For example, to run the tests with debug logging enabled, run `pytest --log-cli-level=DEBUG` from the root of the project.

```shell
poetry run pytest --log-cli-level=DEBUG
```

