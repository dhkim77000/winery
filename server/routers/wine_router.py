from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter


app = FastAPI()


router = APIRouter(
    prefix="/wine",
)





@router.get("/")
async def get_wine_list(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})


@router.get("/{wine_id}")
async def wine_info(wine_id):
    return {'wine_id': wine_id}
# id 형식 길어지는 문제