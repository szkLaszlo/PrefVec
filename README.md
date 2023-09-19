# PrefVeC
This repository contains RL algorithms for highway application with the usage of successor features. 

## Requirements:
SUMO environment can be used with `https://github.com/szkLaszlo/continuousSUMO.git` branch `changes_for_fast_cl`.

##Using the code with Docker:
**Note: First, check if you have right to use docker on your machine.
```docker ps```

Then, the docker needs an environment variable ```WORKSPACE``` defined at your working directory (where continuousSUMO and PrefVeC are cloned)**
If WORKSPACE is missing the training will throw 'ModuleImportError, module not found'

To start the appropriate docker, please run `start_docker.sh` in /docker folder.

It will start a docker container with SUMO, if there is none running. 
If one runs, it will attach to it.

## Basic usage of the code:
To train an agent, inside the docker run the following commands:

```cd $WORKSPACE/PrefVeC```

```python run_trainer.py train```

or run the example file:
```./runner_scripts/example_training.sh```

*Note:
By default the code will log the training in some folder.
You can change it in utils/wrapper.py GLOBAL_LOG_PATH.
Tensorboard and visual logs are available.*