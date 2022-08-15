FROM python:3.11-rc-alpine
WORKDIR /app
ADD ./src /app/
RUN pip3 install openpyxl munch numpy pandas sympy