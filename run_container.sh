if [ ! $( docker ps -a | grep graph | wc -l ) -gt 0 ]; then
    docker run --name graph -itd -v $(pwd)/src:/app luisleo52655/python3.10-graph
fi
if ! docker ps --format '{{.Names}}' | grep -w graph &> /dev/null; then
    docker start graph
fi
git pull origin master
/bin/bash -c "docker exec graph /bin/bash -c 'time python3 -u main.py $@'" 2>&1 | tee log
