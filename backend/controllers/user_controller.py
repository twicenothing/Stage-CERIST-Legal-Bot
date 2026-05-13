from fastapi import APIRouter, Depends,HTTPException
from pydantic import BaseModel
from core.database import get_db
from sqlalchemy.orm import Session
from core.models import User
from services.security import get_password_hash, verify_password, create_access_token, get_admin_user
from typing import Optional
router = APIRouter(prefix="/users")


class UserRequest(BaseModel):
    first_name : str
    last_name : str
    email : str
    password : str
    role : Optional[str] = "utilisateur"

class UserUpdate(BaseModel):
    first_name : Optional[str] = None
    last_name : Optional[str] = None
    email : Optional[str] = None
    password : Optional[str] = None
    role : Optional[str] = None

class LoginRequest(BaseModel):
    email:str
    password:str


@router.post('/login')
def login_user(request: LoginRequest, db: Session = Depends(get_db)):
    existing_user = db.query(User).filter(request.email == User.email).first()
    if not existing_user:
        raise HTTPException(status_code = 401, detail="invalid email or password")

    if not verify_password(request.password, existing_user.hashed_password):
        raise HTTPException(status_code = 401, detail="invalid email or password")
    
    token_data = {"user_id" : existing_user.id, "role": existing_user.role}
    access_token = create_access_token(token_data)

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_name": existing_user.first_name,
        "role": existing_user.role
    }




@router.post('/')
def create_user(request : UserRequest, db : Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    existing_user = db.query(User).filter(request.email == User.email).first()
    if(existing_user):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pwd = get_password_hash(request.password)
    new_user = User(
        first_name = request.first_name, 
        last_name = request.last_name, 
        email = request.email, 
        hashed_password= hashed_pwd,
        role=request.role
    )
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return{
            "message":"User created successfully",
            "user_id": new_user.id,
            "email":new_user.email,
            "role": new_user.role
        }
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        raise HTTPException(status_code = 500, detail= "Internal server error while creating user")


@router.get('/')
def get_all_users(db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    users = db.query(User).filter(User.id != admin.id).all()
    return [
        {
            "id": u.id,
            "first_name": u.first_name,
            "last_name": u.last_name,
            "email": u.email,
            "role": u.role,
            "created_at": u.created_at
        } for u in users
    ]

@router.delete('/{user_id}')
def delete_user(user_id: str, db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    try:
        db.delete(user)
        db.commit()
        return {"message": "User deleted successfully", "id": user_id}
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while deleting user")

@router.put('/{user_id}')
def update_user(user_id: str, request: UserUpdate, db: Session = Depends(get_db), admin: User = Depends(get_admin_user)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
        
    if request.email and request.email != user.email:
        existing_user = db.query(User).filter(request.email == User.email).first()
        if existing_user:
             raise HTTPException(status_code=400, detail="Email already registered")
             
    if request.first_name:
        user.first_name = request.first_name
    if request.last_name:
        user.last_name = request.last_name
    if request.email:
        user.email = request.email
    if request.role:
        user.role = request.role
    if request.password:
        user.hashed_password = get_password_hash(request.password)
        
    try:
        db.commit()
        db.refresh(user)
        return {
            "message": "User updated successfully",
            "id": user.id,
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role
        }
    except Exception as e:
        db.rollback()
        print(f"Database error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error while updating user")