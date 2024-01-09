#%%
from fastapi.templating import Jinja2Templates
from fastapi import FastAPI,Request
import numpy as  np

templates = Jinja2Templates(directory="templates")


#%%
#%% mdoel yükle

from joblib import load
filename='my_second_saved_model'

uploadedModel=load(filename)

#%%



app = FastAPI()


@app.get("/")

async def read_root(request:Request):
    return  templates.TemplateResponse("base.html", {"request": request})

 
    
    
@app.get("/predict/")
async def make_prediction(request:Request,L1:float,W1:float,L2:float,W2:float):
    
    testData=np.array([L1,L2,W1,W2]).reshape(-1,4)
    probabilities=uploadedModel.predict_proba(testData)[0]
    predicted=np.argmax(probabilities)
    probality=probabilities[predicted]
    
    
    return  templates.TemplateResponse("prediction.html", {"request": request,' probabilities': probabilities,
                                                           'predicted':predicted,
                                                           'probality':probality})

#%%