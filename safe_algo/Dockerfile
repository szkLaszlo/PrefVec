FROM ubuntu:22.04

RUN echo "fs.inotify.max_user_watches = 524288" >> /etc/sysctl.conf
RUN sysctl -p --system

# Gnome-terminal and locales
ENV LANG=en_US.UTF-8
RUN apt-get update -qq && apt-get install -qy gnome-terminal libcanberra-gtk-module libcanberra-gtk3-module locales
RUN echo 'LANG=en_US.UTF-8' > '/etc/default/locale' && \
    locale-gen --lang en_US.UTF-8 && \
    dpkg-reconfigure --frontend=noninteractive locales && \
    update-locale LANG=$LANG

# Install make and compilers and extra stuff
RUN DEBIAN_FRONTEND=noninteractive apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -qy \
    build-essential autoconf automake \
    sudo vim nano git curl wget tmux \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
    gcc x11-apps git openssh-client libfontconfig1 \
    emacs tcpdump telnet byacc flex \
    iproute2 gdbserver less bison valgrind \
    libxtst-dev libxext-dev libxrender-dev libfreetype6-dev \
    openssh-server cmake gdb build-essential clang llvm lldb && \
    apt-get clean -qq && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /tmp/*

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN echo "PATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/environment
RUN echo "PYTHONPATH=/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" >> /etc/environment
RUN echo $PYTHONPATH
RUN mkdir -p ~/miniconda3
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh --no-check-certificate -O ~/miniconda3/miniconda.sh
RUN bash ~/miniconda3/miniconda.sh -b -u -p /opt/conda
RUN rm -rf ~/miniconda3/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

COPY ./omnisafe_zjy /omnisafe_zjy

RUN conda init 
RUN conda install -c anaconda -y python=3.8

WORKDIR /omnisafe_zjy
RUN pip install -e .
RUN pip install highway_env opencv-python

ENTRYPOINT /entry.sh
