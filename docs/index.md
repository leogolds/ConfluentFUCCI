## ConfluentFUCCI

A suite of tools for analyzing large scale confluent FUCCI experiments

### Showcase
![Example 1](figures/fig1.png)
![Example 2](figures/fig2.png)

### Installation
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

#### Using docker-compose
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
