# Run this file for the sumo container to start.
export UID
export USER
export GID

docker_list_raw=`docker ps --filter "name=saferl" -q`
docker_list=($docker_list_raw)

if [ ${#docker_list[@]} -eq 0 ]; then
        echo "We have no running docker container, so we start one."
        docker compose -f ./docker-compose.yml run --name saferl --rm saferl
else
        echo "We are entering the only running docker container."
	docker exec -it -e HOST_USER=$USER -e HOST_UID=$UID -e UID=$UID -e GID=$GID -u 0 ${docker_list[0]} /user_entry.sh
fi
