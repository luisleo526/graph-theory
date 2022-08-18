FROM python:3.10-bullseye
WORKDIR /app
RUN apt update && apt upgrade -y && apt install sshpass git -y
RUN pip3 install openpyxl munch numpy pandas sympy numba
RUN mkdir /root/.ssh
