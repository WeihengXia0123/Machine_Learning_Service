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

    # git commit and push
    subprocess.run(["git", "commit", "-m", f"test: add new dataset"],stdout=subprocess.PIPE)
    # subprocess.run(["git", "push"],stdout=subprocess.PIPE)

    # dvc push to remote data storage (optional)
    # FIXME: unsupported pickle protocol: 5 
    # (only happens when using python 3.6, because DVC uses pickle protocol 5 which is only suppported in python 3.8+)
    # (however, I use python 3.6 because my GPU nvidia-smi is v384 and I must use python 3.6 to use CUDATOOLKIT 9.0)
    # (a better solution is to use Ubuntu 18.04 for this project, if you want to use GPU for training)
    
    # subprocess.run(["dvc", "push"],stdout=subprocess.PIPE)

    return f"uploading data: {data_path}"


@app.post("/start_training/")
async def start_training():
    """
    A POST request to start the training with the latest data version
    """
    # To get the latest data, you can do either `dvc pull` to sync with remote storage
    # or, you can do `git checkout` to the latest git commitID, then `dvc checkout` to sync with local storage
    # subprocess.run(["dvc", "pull"])

    # start the training
    train_epoch = 2
    pipeline.train(train_epoch)
    app.last_epoch_number += train_epoch
    
    # dvc add ckpt/
    subprocess.run(["dvc", "add", "ckpt/"], stdout=subprocess.PIPE)
    subprocess.run(["git", "add", "ckpt.dvc"], stdout=subprocess.PIPE)        
    subprocess.run(["git", "commit", "-m", f"test: add new ckpt"],stdout=subprocess.PIPE)
    # subprocess.run(["git", "push"],stdout=subprocess.PIPE)
    # subprocess.run(["dvc", "push"],stdout=subprocess.PIPE)

    print("finished training with epochs: ", app.last_epoch_number)
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
    image, prediction, label = pipeline.infer(data_path)
    label = int(label[0])
    prediction = int(prediction[0])
    print("prediction result: ", prediction)
    print("groundtruth label: ", label)
    # plt.imshow(image)
    # plt.show()
    return [{"prediction digit": str(prediction)}, {"label", label}]
