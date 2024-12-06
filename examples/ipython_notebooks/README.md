# Introduction

Welcome to the example notebooks.  Here is a quick guide to
getting the data and installing BluePy to use these example notebooks.


## Using BB5

The recommended way is to use the examples with BB5. In order to do so, you should log-in to BB5 and follow the instructions below.

### Create jupyter kernel

In order to use a jupyter kernel with bluepy installed, you need to create a new kernel with python 3.7 (python 3.8 is incompatible with `bluepy` for the moment):

    /gpfs/bbp.cscs.ch/apps/tools/jupyter/create-jupyterlab-venv ~/BluepyKernel viz/latest ffmpeg archive/2020-09 python/3.7.4

You can name the kernel `BluepyKernel` or whatever you like.

### Installing BluePY

The next step is to install bluepy in that kernel which you can do with:

    ~/BluepyKernel/bin/pip install bluepy[all] -i https://bbpteam.epfl.ch/repository/devpi/simple/

Replace `BluepyKernel` with the name you have chosen for the kernel. 

### Start the jupyter notebook

Next, you create a local checkout of the repository

    git clone git@bbpgitlab.epfl.ch:nse/bluepy.git

and then you can start a jupyter instance on BB5 via [https://ood.bbp.epfl.ch/](https://ood.bbp.epfl.ch/), and navigate to the installed `bluepy` folder, and load and run the notebooks.

## Using Ubuntu

You can also use the examples notebooks on a local ubuntu installation. For that you have to make sure to use python3.7 and that `libpython3.7` is installed:

    sudo apt-get install libpython3.7
   
### Create jupyter kernel

Also here you need to create a virtual environment with some packages installed which will become the kernel for jupyter:

    virtualenv -p python3.7 venv_bluepy  # or any name
	. venv_bluepy/bin/activate
	pip install -U pip setuptools
	pip install -i https://bbpteam.epfl.ch/repository/devpi/simple/ bluepy[all]
	pip install ipython jupyter
	ipython kernel install --user --name=venv_bluepy  # name is the folder name of your venv
	
### Start the jupyter notebook

Next, you create a local checkout of the repository

    git clone git@bbpgitlab.epfl.ch:nse/bluepy.git

and you can start a local jupyter instance 

    jupyter notebook
    
This will open a browser, in which you can navigate to the installed `bluepy` folder, and load and run the notebooks.
