from fastapi import FastAPI
from fastapi import FastAPI, Form, Request

import uvicorn



from routers import user_router, home_router ,wine_router, mbti_router
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router.router)
app.include_router(home_router.router)
app.include_router(wine_router.router)
app.include_router(mbti_router.router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)