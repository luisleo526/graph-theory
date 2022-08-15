FROM python:3.11.0rc1-bullseye
WORKDIR /app
ADD ./src /app/
RUN pip3 install openpyxl munch numpy pandas sympy