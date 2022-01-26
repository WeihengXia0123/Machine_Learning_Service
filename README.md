# Machine_Learning_Service
@brief \
@author: Weiheng Xia

(TODO: I will write in detail when I have more time after this week's 996 work~;)

## Code structure

## How to run the code
```bash
sudo docker build . -f dockerfile -t mls:v1
sudo docker run -p 80:80 mls:v1
# Then open 127.0.0.1:80/docs in your browser to test the functions

# If you want to get inside the docker container, do the following:
sudo docker exec -it [containerID] bash
```

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
