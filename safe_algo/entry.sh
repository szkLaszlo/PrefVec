#! /bin/bash

ulimit -c unlimited
echo "Please leave this terminal open, go to a new terminal window and run ./start_docker.sh"
su "$@"