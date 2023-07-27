from fastapi import FastAPI

import uvicorn



from routers import user_router, home_router ,wine_router
from fastapi.middleware.cors import CORSMiddleware
from routers import recommend_router


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
app.include_router(recommend_router.router)

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=30005)