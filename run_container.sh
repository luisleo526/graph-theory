if [ ! $( docker ps -a | grep graph | wc -l ) -gt 0 ]; then
    #if [[ "$(docker images -q graph:latest 2> /dev/null)" == "" ]]; then
    #    echo "FROM python:3.11.0rc1-bullseye" >> Dockerfile
    #    echo "WORKDIR /app" >> Dockerfile
    #    echo "RUN pip3 install openpyxl munch numpy pandas sympy" >> Dockerfile
    #    docker build -t graph .
    #fi
    docker run --name graph -itd -v $(pwd)/src:/app luisleo52655/python3.11:
fi
docker exec graph /bin/bash -c "time python3 main.py $@"
