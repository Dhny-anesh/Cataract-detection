from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
import cv2
#from tensorflow.keras.models import load_model
from keras.models import load_model

# Load the trained model
model = load_model('vgg19.h5')

#Load and preprocess a new image
def preprocess_image(image_path, target_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to [0, 1]
    return image


app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount the static folder to serve CSS files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/submit", response_class=HTMLResponse)
async def submit_form(request: Request):
    form_data = await request.form()
    '''
    # Path to the new image you want to classify
    new_image_path = '/content/Fundus_photograph_of_normal_left_eye.jpg'
    # Preprocess the new image
    new_image = preprocess_image(new_image_path)
    # Make a prediction
    prediction = model.predict(np.expand_dims(new_image, axis=0))

    # Interpret the prediction
    if prediction[0][0] > 0.5:
        result = "Cataract"
    else:
        result = "Normal"'''
    #return templates.TemplateResponse("result.html", {"request": request, "url": form_data["url"], "prediction": result})
    print(form_data)
    return templates.TemplateResponse("result.html", {"request": request})



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="8000")