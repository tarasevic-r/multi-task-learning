# Multi-task-learning

## Overview
Multi-task learning project, where the main task has very little data, 
but the auxiliary task has a lot of data. The expectation is that the data 
from the second task will generalise the model well and then a small amount 
of data from the first task will be sufficient to train the network and avoid 
the overfiting problem. The main task is binary classification, the auxiliary 
task is regression.


## Prerequisites 
1. Framework: Python 3.8
2. Operating system: Ubuntu (this app wasn't tested on any different OS)

### Virtualenv

To install it, run:
```
# Get the latest version
pip install virtualenv
```

To create a virtual environemnt in the local file system the syntax is:
```
virtualenv --python <python version> <name of environment> 
```

For example, to create an empty Python 3.6 environment: 
```
virtualenv --python 3.8 mtl_project
```

## Activating and populating the environment with packages

To activate the created empty environment, use the command: 
```
source mtl_project/bin/activate
```

### Install requirments
Installing the necesary packages:
```
pip install -r requirements.txt
```

## Run model training
From command line in a root project directory
```
python3 -m run_training
```
Another way is to open `train_model.ipynb` that is located in *./notebooks* directory and run that notebook code.

## Inference
For inference, a streamlit application is created where input values are given and a prediction 
is returned by clicking "predict".  
**Note: the streamlit application uses the last model you trained**

Run from command line in a root project directory
```
streamlit run app.py 
```

![My Image](inference.png)