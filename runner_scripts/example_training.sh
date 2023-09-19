# Examples for each env

#######GridWorld
# DQN
python run_trainer.py train --env_model grid --model_version q --model_train_type q --preference_type default --default_w -1000 1 --comment grid --seed 15348 &
# PrefVeC_S
python run_trainer.py train --env_model grid --comment grid --seed 15348 &
# PrefVeC_D
python run_trainer.py train --env_model grid --model_train_type dynamic --comment grid --seed 15348

#######Merge
# DQN
python run_trainer.py train --env_model merge --model_version q --model_train_type q --preference_type default --default_w -1000 1 --comment basic_merge --seed 15348 &
# PrefVeC_S
python run_trainer.py train --env_model merge --comment basic_merge --seed 15348 &
# PrefVeC_D
python run_trainer.py train --env_model merge --model_train_type dynamic --comment basic_merge --seed 15348

## for the zero initial speed add the flag --use_random_speed 0