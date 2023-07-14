from fastapi import APIRouter
from fastapi import Depends
from sqlalchemy.orm import Session
from starlette import status
from psycopg2.extensions import connection
from fastapi import FastAPI, Form, Request
from fastapi.templating import Jinja2Templates

from crud import create_user
from schema import UserCreate
from database import get_db, get_conn
import pdb , os
import uvicorn
from fastapi.responses import RedirectResponse, HTMLResponse
from crud import create_user, get_user, verify_password , add_mbti_feature
app = FastAPI()
templates = Jinja2Templates(directory=os.path.dirname(os.getcwd())+'/templates')

router = APIRouter(
    prefix="/home",
)

# 홈 화면
@router.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    
    return templates.TemplateResponse("success.html", {"request": request})


@router.get("/recommendations")
async def get_register_form(request: Request):
    return templates.TemplateResponse('recommendations_from.html', context={'request': request})

@router.post("/recommendations", status_code=status.HTTP_303_SEE_OTHER)
async def post_item_by_mbti_id(request: Request, email: str = Form(...), mbti_id: int = Form(...), db: connection = Depends(get_conn)):
    # Perform validation on email and mbti_id if needed

    # Call add_mbti_feature function to save item
    result = await add_mbti_feature(email, mbti_id, db)
    return result



