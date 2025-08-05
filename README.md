# Homomesy

This repository contains code to visualize and analyze homomesic behavior of different sets and statistics.

## How to run

The code is using sage, a python library for mathematics.

First, you need to install docker. You can find the instructions [here](https://docs.docker.com/get-docker/).

Then you need to pull the sage image:

```bash
docker pull sagemath/sagemath
```

Then start the jupiter notebook server:

```bash
docker run --name homomessy_notebook_server -p8888:8888  -v "${PWD}/homomesy":/home/sage/homomesy -v "${PWD}/out":/home/sage/out sagemath/sagemath:latest sage-jupyter
```

After that, an url will be printed that can be used to connect to the jupiter notebook server using any IDE. 
Make sure to use the Python kernel and **not** the Sage kernel, as the code is written in Python.

## How to use

the folder `out` is mounted to the container, so if you want to save some plots, you have to save them in the `out` folder, otherwise they will not be visible to you.
For example `plt.savefig("out/plot.png")` will save the plot in the `out` folder.



