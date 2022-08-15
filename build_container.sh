if [[ "$(docker images -q myimage:mytag 2> /dev/null)" == "" ]]; then
        echo "FROM python:3.11.0rc1-bullseye" >> Dockerfile
        echo "WORKDIR /app" >> Dockerfile
        echo "RUN pip3 install openpyxl munch numpy pandas sympy" >> Dockerfile
        docker build -t graph .
fi
docker run --name $1 -itd -v $(pwd)/src:/app graph