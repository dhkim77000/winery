from fastapi import FastAPI, Form, Request, Response
from fastapi import APIRouter

app = FastAPI()

router = APIRouter(
    prefix="/wine",
)


@router.get("/")
async def get_login_form(request: Request):
    return templates.TemplateResponse('login_form.html', context={'request': request})
