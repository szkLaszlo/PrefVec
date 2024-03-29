# PrefVeC
This repository contains RL algorithms for highway application with the usage of successor features. 

## Requirements:
Set your ```WORKSPACE``` environment variable to the path of your working directory.

Change to the directory where you want to clone the repository:
```cd $WORKSPACE```

SUMO environment can be used with `https://github.com/szkLaszlo/continuousSUMO.git` branch `changes_for_fast_cl`.

```
git clone https://github.com/szkLaszlo/continuousSUMO.git 
cd continuousSUMO
git checkout changes_for_fast_cl
cd ..
 ```
Clone PrefVeC repository into your WORKSPACE:

```git clone https://github.com/szkLaszlo/PrefVeC.git  ```

For the intersection-env you need to clone `https://github.com/szkLaszlo/rl-agents.git` main branch into a 'rlagents' folder in your WORKSPACE.

```git clone https://github.com/szkLaszlo/rl-agents.git rlagents```


##Using the code with Docker:
**Note: First, check if you have right to use docker on your machine.
```docker ps```

Then, the docker needs an environment variable ```WORKSPACE``` defined at your working directory (where continuousSUMO and PrefVeC are cloned)**
If WORKSPACE is missing the training will throw 'ModuleImportError, module not found'

To start the appropriate docker, please run `start_docker.sh` in /docker folder.

```
cd $WORKSPACE/PrefVeC/docker
./start_docker.sh
```

It will start a docker container with SUMO, if there is none running. 
If one runs, it will attach to it.

## Basic usage of the code:
To train an agent, inside the docker run the following commands:

```
cd $WORKSPACE/PrefVeC
python run_trainer.py train
```

or run the example file:
```
./example/example_training.sh
```

*Note:
By default the code will log the training in some folder.
You can change it in utils/wrapper.py GLOBAL_LOG_PATH.
Tensorboard and visual logs are available.*

## Recreating the safe baseline results:
For safeRL comparison check out the safeRL_baseline branch and see README.md in the safe_algo folder.
## Recreating the PPO and SAC results:
Similarly, for PPO and SAC comparison check out the safeRL_baseline branch.
You can run the original container and inside the container pip install the req_for_rllib.txt. Then you can run the rllib_examples.sh. 
If pip install does fail maching the versions, restart the container, try with ray==2.3.1 in the req_for_rllib.txt then install ray==2.4.0.

## Reward component ablation:
To recreate the results of the reward component ablation, you need to run './experiments.sh'  inside the docker on the main branch.
