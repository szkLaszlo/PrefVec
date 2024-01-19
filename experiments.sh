# Merge environment with different weights

# PrefVeC - dynamic - standard weights
python run_trainer.py train --seed 10 --comment s10_1M &
python run_trainer.py train --seed 5698 --comment s5698_1M
# PrefVeC - dynamic - true reward = [-1000, 1]
python run_trainer.py train --seed 10 --eval_w 1 -1000 1 --w 6 0 1 0 1 0 1 0 1 0 1 -1000 1    --comment s10_1k &
python run_trainer.py train --seed 5698 --eval_w 1 -1000 1 --w 6 0 1 0 1 0 1 0 1 0 1 -1000 1 --comment s5698_1k
# PrefVeC - dynamic - true reward = [-100, 1]
python run_trainer.py train --seed 10 --eval_w 1 -100 --w 6 0 1 0 1 0 1 0 1 0 1 -100 1 1 --comment s10_100 &
python run_trainer.py train --seed 5698 --eval_w 1 -100 1 --w 6 0 1 0 1 0 1 0 1 0 1 -100 1 --comment s5698_100

# DQN - standard weights
python run_trainer.py train --seed 10 --model_version q --model_train_type q --preference_type default --default_w -1000000 1 --comment s10_1M &
python run_trainer.py train --seed 5698 --model_version q --model_train_type q --preference_type default --default_w -1000000 1 --comment s5698_1M
# DQN - true reward = [-1000, 1]
python run_trainer.py train --seed 10 --model_version q --model_train_type q --preference_type default --default_w -1000 1 --comment s10_1k &
python run_trainer.py train --seed 5698 --model_version q --model_train_type q --preference_type default --default_w -1000 1 --comment s5698_1k
# DQN - true reward = [-100, 1]
python run_trainer.py train --seed 10 --model_version q --model_train_type q --preference_type default --default_w -100 1 --comment s10_100 &
python run_trainer.py train --seed 5698 --model_version q --model_train_type q --preference_type default --default_w -100 1 --comment s5698_100

# PrefVeC - dynamic - standard weights - no new PER
python run_trainer.py train --seed 10 --weight_loss_with_sf 0 --comment s10_1M_noPER
