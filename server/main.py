from fastapi import FastAPI
from fastapi import FastAPI, Form, Request

import uvicorn
from server.routers import user_router

app = FastAPI()
app.include_router(user_router.router)


if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)