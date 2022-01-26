# Machine_Learning_Service
(TODO: I will write README in detail, when I have more time after this week's 996 work~;)

## Code structure

## How to run the code
```bash
sudo docker build . -f dockerfile -t mls:v1
sudo docker run -p 80:80 mls:v1
# If you want to get inside the docker container, do the following:
# sudo docker exec -it [containerID] bash
```

## Test cases
open 127.0.0.1:80/docs in your browser to test the functions:
1. POST request to upload data, enter the path for new data that you want to add, \
   example: **/workspace/new_dataset/73000_73257.mat**
2. POST request to train the network, enter the num of epochs that you want to train,\
   example: **10**
3. GET request to get the current training status, total epochs + validation accuracy
4. POST request to do inference on a data,\
   example: **/workspace/new_dataset/41000_42000.mat**

## Dependencies
See requirements.txt and environment.yaml

## Docker
1. Prepare conda/pip requirements in git repo
2. Dockerfile: install git, conda
3. Import conda/pip requirements (Pip, Python 3.8, FastAPI, DVC)
4. Copy current folder code into docker, without datasets
5. git init / dvc init
----> I can git/dvc push to remote storage, like my Google Drive, but it requires my credentials, which is not safe to put on the git repo. Therefore, I decided to go local commit instead of remote storage (but I can achieve this, by putting credentials into docker)
6. RUN uvicorn app
