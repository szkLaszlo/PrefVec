python rllib_examples.py --env_model grid --use_step_reward 0 --default_w -1000 1 --comment s15348_nostepr_1k --seed 15348 &
python rllib_examples.py --env_model grid --use_step_reward 1 --default_w -1000 1 --comment s15348_stepr_1k --seed 15348
python rllib_examples.py --env_model grid --use_step_reward 0 --default_w -1000 1 --comment s15_nostepr_1k --seed 15 &
python rllib_examples.py --env_model grid --use_step_reward 1 --default_w -1000 1 --comment s15_stepr_1k --seed 15

# intersection env
python rllib_examples.py --env_model intersection --default_w -1000 1 1 --comment s15348_1k --seed 15348 &
python rllib_examples.py --env_model intersection --default_w -1000 1 1 --comment s15_1k --seed 15

