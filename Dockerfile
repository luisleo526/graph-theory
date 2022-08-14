FROM ubuntu:20.04
WORKDIR /app
RUN apt update && apt upgrade -y
RUN apt-get update \
    &&  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata
RUN TZ=Asia/Taipei \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && dpkg-reconfigure -f noninteractive tzdata 
RUN apt install software-properties-common git -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3.10 python3-pip -y
RUN apt install sshpass -y
RUN mkdir /root/.ssh
RUN touch /root/.ssh/known_hosts
RUN ssh-keyscan github.com >> ~/.ssh/known_hosts
RUN ssh-keyscan 140.112.26.25 >> ~/.ssh/known_hosts
RUN git clone https://github.com/luisleo526/graph-theory.git /app
RUN pip3 install openpyxl munch numpy pandas sympy