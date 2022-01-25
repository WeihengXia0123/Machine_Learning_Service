from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import subprocess, os
import model.train as train
import matplotlib.pyplot as plt
app = FastAPI()

# global variables
app.last_epoch_number = 0

# import model_pipeline: train, validation
pipeline = train.model_pipeline()

@app.post("/upload_data/{data_path: path}")
async def upload_data(data_path: str):
    """
    A POST request endpoint to upload new training data (assume the data cannot be repeated, so there is no need to
    check for repeated data), combines it with existing data and versions the dataset. You are encouraged to use
    third party tools to version models and data (e.g: https://dvc.org)
    """
    # upload data to the Dataset folder
    subprocess.run(["cp", data_path, "dataset/"])

    # dvc add and commit new dataset
    subprocess.run(["dvc", "add", "dataset/"], stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "dataset.dvc"], stdout=subprocess.PIPE)
    subprocess.run(["dvc", "add", "ckpt/"], stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "ckpt.dvc"], stdout=subprocess.PIPE)

    # git commit and push
    subprocess.run(["git", "commit", "-m", f"test: add new dataset\n{data_path}"],stdout=subprocess.PIPE)
    subprocess.run(["git", "push"],stdout=subprocess.PIPE)

    # dvc push to remote data storage (optional)
    # FIXME: unsupported pickle protocol: 5 
    # (only happens when using python 3.6, because DVC uses pickle protocol 5 which is only suppported in python 3.8+)
    # (however, I use python 3.6 because my GPU nvidia-smi is v384 and I must use python 3.6 to use CUDATOOLKIT 9.0)
    # (a better solution is to use Ubuntu 18.04 for this project, if you want to use GPU for training)
    subprocess.run(["dvc", "push"])

    return f"uploading data: {data_path}"


@app.post("/start_training/")
async def start_training():
    """
    A POST request to start the training with the latest data version
    """
    # dvc pull to get the latest data
    subprocess.run(["dvc", "pull", "-y"])

    # start the training
    train_epoch = 20
    epoch_num = pipeline.train(train_epoch)
    app.last_epoch_number += epoch_num

    return "finished training"


@app.get("/get_status")
async def get_status():
    """
    A GET request to retrieve the status of the training (last epoch number, and validation metrics).
    """
    # get validation result
    print("VALIDATION RESULTS:")
    validation_result = pipeline.valid()
    validation_result.append({"curr_epoch": app.last_epoch_number})
    return validation_result


@app.post("/result/{data_path: path}")
async def result(data_path : str):
    """
    An inference POST endpoint for a single data instance that returns the latest model results on it.
    """
    print("INFERENCE RESULTS:")
    data_path = "/home/weiheng/Code/Coding_Interview/Peter_Park/Machine_Learning_Service/dataset/0_1000.mat"
    image, prediction = pipeline.infer(data_path)
    print("prediction result: ", prediction)
    # plt.imshow(image)
    # plt.show()
    return {"prediction digit": str(prediction)}
