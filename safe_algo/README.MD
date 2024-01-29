To run the safeRL experiment with our environment, you need to use the fork from the original omnisafe repo. If you use the container we provide, it has the code already installed. If you want to install it yourself, you can do so by cloning szklaszlo/omnisafe, check out the dev-discrete and install the package with pip.

Note: make sure you modify the docker-compose.yml based on your preferences. Especially to map the volumes correctly.

The experiments were run inside the container with python safe_experiments.py. The file contains 'cartpole' but in the container, we use the modified code from szklaszlo/omnisafe, so our highway gym is running instead of the original one.
