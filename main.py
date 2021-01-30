
#import fastapi #this does not work

from fastapi import FastAPI   #this works
import uvicorn


app = FastAPI()         #calling FastAPI class from fastapi


@app.get('/urlname')         # "urlname" added after / for unique address


async def root (x=2 , y=9) :
 return x+y 

def root2 (m=2 , n=9) :
 return m+n

#print (x+y)

#if __name__  == "__main__":
 #   uvicorn.run(app,port=8000,host='0.0.0.0')
 