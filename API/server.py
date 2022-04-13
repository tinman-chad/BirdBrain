import os
from os.path import exists
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response, HTMLResponse
import BirdBrain
from zipfile import ZipFile
from PIL import Image
import warnings
import io
from functools import lru_cache 

warnings.filterwarnings("ignore")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="BirdBrain API",
        version="0.1.1",
        description="This api allows the user to upload an image and uses Tensorflow's Object Detection API to find birds in the image and users a custom trained image classifcation model based upon EfficentNetV2L using Keras to classify the bird species of any birds found in the image.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

@lru_cache()
def get_models():
    detection_fn, detection_lbl, class_lbl, class_model = BirdBrain.InitModels()
    return detection_fn, detection_lbl, class_lbl, class_model

@app.on_event("startup")
async def startup_event():
    get_models()

@app.get("/health")
async def health():
    """Health check endpoint with checks to make sure model is loaded."""
    status = 'available'
    detection_fn, detection_lbl, class_lbl, class_model = get_models()
    if detection_fn == None or detection_lbl == None or class_lbl == None or class_model == None:
        return Response(content={'status': 'un' + status}, status=500)

    return Response(content={'status': status}, status=200)

@app.get("/ping")
async def ping():
    """Keep alive"""
    return "pong"

@app.get("/")
async def root():
    return HTMLResponse(status_code=200, content="""
    <html><head><title>BirdBrain V. 0.1.1</title></head><body>
        <h1>BirdBrain</h1>
        <div>Select a file and then choose to find the birds.<br>
        Click the JSON Data for the full details of what was found.<br>
        Click the Image with Boxes to show boudning boxes around the birds.<br></div>
        <form method=post enctype='multipart/form-data'>
            <input type=file name=image>
            <input type=submit value='JSON Data'>
            <input type=button value='Image with Boxes' onclick="javascript: this.form.action='/ShowBounds'; this.form.submit()">
        </form>
    </body></html>
    """)

@app.post("/")
async def ProcessImage(image: UploadFile):
    """
        Processing an image for the bird species and confidence scores returned as a json object with the bounding box points.
    """
    if image.content_type == "image/jpeg":
        im_np = BirdBrain.load_image_into_numpy_array_bytes(await image.read())
        detection_fn, detection_lbl, class_lbl, class_model = get_models()
        ret = BirdBrain.find(im_np, detection_fn, detection_lbl, class_lbl, class_model)
        return JSONResponse(status_code=200, content=ret['Predictions'])
    
    return JSONResponse(status_code=406, content={"message":"Invalid file type."})

@app.post("/ShowBounds")
async def ShowBounds(image: UploadFile):
    """
        Processing an image for the bird species and confidence scores returned as an image with the information drawn on it.
    """
    if image.content_type == "image/jpeg":
        im_np = BirdBrain.load_image_into_numpy_array_bytes(await image.read())
        detection_fn, detection_lbl, class_lbl, class_model = get_models()
        ret = BirdBrain.find(im_np, detection_fn, detection_lbl, class_lbl, class_model)
        im = Image.fromarray(ret['ImageWithBoxes'])

        img_byte_arr = io.BytesIO()
        im.save(img_byte_arr, format='JPEG')
        return Response(content=img_byte_arr.getvalue(), status_code=200)

        #return StreamingResponse(Image.fromarray(ret['ImageWithBoxes']).getdata(), media_type="image/jpeg") 
    
    return JSONResponse(status_code=406, content={"message":"Invalid file type."})

@app.post("/UploadNewModelPackage")
async def UploadNewModelPackage(file: UploadFile):
    """
        Upload a new model package file as a zip file containing the hdf5 and pkl model files and labels.txt as the new staged model package.
    """
    if file.content_type == "application/zip":
        with open(BirdBrain.PROJECT_DATA_PATH + f'Stage/{file.filename}', 'wb') as file:
            file.write(await file.read())
        
        zipfile.extractall(BirdBrain.PROJECT_DATA_PATH + f'Stage/{file.filename}')

        os.remove(BirdBrain.PROJECT_DATA_PATH + f'Stage/{file.filename}')

        return JSONResponse(status_code=204, content={"message":"File Staged."})
    
    return JSONResponse(status_code=406, content={"message":"Invalid file type."})

@app.post('/CompareModels')
async def CompareModels(image: UploadFile):
    """
        Process the supplied image with the staged model package and the current released model package, returns the json results of both packages in a dictionary containing two keys [Stage, Production].
    """
    if image.content_type == "image/jpeg":
        im_np = BirdBrain.load_image_into_numpy_array_bytes(await image.read())
        ret = BirdBrain.find(im_np)
        compared = {'Production': ret['Predictions']}
        currentPath = BirdBrain.PROJECT_MODEL_PATH
        BirdBrain.PROJECT_MODEL_PATH = BirdBrain.PROJECT_DATA_PATH + "Stage/"
        ret2 = BirdBrain.find(im_np)
        compared['Stage'] = ret2['Predictions']
        BirdBrain.PROJECT_MODEL_PATH = currentPath
        return JSONResponse(status_code=200, content=compared)

    return JSONResponse(status_code=406, content={"message":"Invalid file type."})

@app.put('/PromoteModel')
async def PromoteModel(ModelName: str):
    """
        Replace the current Production model package with the current Stage model package.
    """
    if not exists(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.hdf5"):
        return JSONResponse(status_code=406, content={"message":"Model not found."})

    try:
        if exists(BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/model.hdf5"):
            #remove current model
            os.remove(BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/model.hdf5")
            os.remove(BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/model.pkl")
            os.remove(BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/labels.txt")
        #move new model
        os.rename(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.hdf5", BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/model.hdf5")
        os.rename(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.pkl", BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/model.pkl")
        os.rename(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/labels.txt", BirdBrain.PROJECT_MODEL_PATH + f"{ModelName}/labels.txt")

    except:
        return JSONResponse(status_code=500, content={"message":'Error, Please validate the model works or replace it again.'})
    BirdBrain.PROJECT_MODEL_PATH
    return JSONResponse(status_code=201, content={"message":'Promoted'})

@app.delete('/DropModel')
async def DropModel(ModelName: str):
    """
        Removes the current Stage model package if it does not compare well you wouldn't want it to accidentally go into production.
    """
    try:
        if exists(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.hdf5"):
            os.remove(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.hdf5")
            os.remove(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/model.pkl")
            os.remove(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}/labels.txt")
            os.rmdir(BirdBrain.PROJECT_DATA_PATH + f"Stage/{ModelName}")
            return JSONResponse(status_code=201, content={"message":'Removed'})
    except:
        return JSONResponse(status_code=500, content={"message":'Error, Please validate the modelwas removed.'})

    return JSONResponse(status_code=404, content={"message":"Model not found."})
