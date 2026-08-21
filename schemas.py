from pydantic import BaseModel , EmailStr

class UserLogin(BaseModel):
    username: str
    password: str 

class Token(BaseModel):
    access_token: str 
    token_type: str = "bearer"

class UserData(BaseModel):
    username: str 
    tenant_id: str 