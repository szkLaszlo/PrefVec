python rllib_examples.py train --env_model grid --exit_threshold 1 --use_step_reward 0 --default_w 1 -1000 1 --comment s15348_nostepr_1k --seed 15348 &
python rllib_examples.py train --env_model grid --exit_threshold 1 --use_step_reward 1 --default_w 1 -1000 1 --comment s15348_stepr_1k --seed 15348
python rllib_examples.py train --env_model grid --exit_threshold 1 --use_step_reward 0 --default_w 1 -1000 1 --comment s15_nostepr_1k --seed 15 &
python rllib_examples.py train --env_model grid --exit_threshold 1 --use_step_reward 1 --default_w 1 -1000 1 --comment s15_stepr_1k --seed 15

# intersection env
python rllib_examples.py train --env_model intersection --default_w 1 -1000 1 1 --comment s15348_1k --seed 15348 &
python rllib_examples.py train --env_model intersection --default_w 1 -1000 1 1 --comment s15_1k --seed 15

