if [ ! "$(docker ps -q -f name=graph)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=graph)" ]; then
            if [[ "$(docker images -q graph:latest 2> /dev/null)" == "" ]]; then
                echo "FROM python:3.11.0rc1-bullseye" >> Dockerfile
                echo "WORKDIR /app" >> Dockerfile
                echo "RUN pip3 install openpyxl munch numpy pandas sympy" >> Dockerfile
                docker build -t graph .
            fi
            docker run --name graph -itd -v $(pwd)/src:/app graph
    fi
fi
docker exec graph /bin/bash -c "time python3 main.py -n $1 -t $2"