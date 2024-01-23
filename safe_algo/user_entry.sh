#! /bin/bash
echo "$HOST_USER:x:$HOST_UID:" >> /etc/group
echo "docker:x:999:$HOST_USER" >> /etc/group
useradd -u $HOST_UID -g $HOST_UID -d /home/$HOST_USER -s /bin/bash -M $HOST_USER
sed -i "s|^PYTHONPATH.*|&:$WORKSPACE|" /etc/environment
cd /home/$HOST_USER
su $HOST_USER "$@"
