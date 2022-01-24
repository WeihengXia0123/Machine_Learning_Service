from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess, os

app = FastAPI()

# global variables
app.i = 0

@app.post("/upload_data/{data_path: path}")
async def upload_data(data_path: str):
    """
    A POST request endpoint to upload new training data (assume the data cannot be repeated, so there is no need to
    check for repeated data), combines it with existing data and versions the dataset. You are encouraged to use
    third party tools to version models and data (e.g: https://dvc.org)
    """
    # upload data to the Dataset folder
    subprocess.run(["cp", data_path, "dataset"])

    # dvc add and commit new dataset
    

    # push to remote data storage

    return {"uploading data": data_path}


@app.post("/start_training/")
async def start_training():
    """
    A POST request to start the training with the latest data version
    """
    return "start training.. "


@app.get("/get_status", response_class=HTMLResponse)
async def get_status():
    """
    A GET request to retrieve the status of the training (last epoch number, and validation metrics).
    """
    return f"current epoch is {app.i}"


@app.post("/result/{instance_num}")
async def result(instance_num):
    """
    An inference POST endpoint for a single data instance that returns the latest model results on it.
    """
    return "the result of instance is .."
