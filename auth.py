import os 
from datetime import datetime , timedelta , timezone 
from jose import jwt , JWTError 
from passlib.context import CryptContext
from fastapi import Depends , HTTPException , status 
from fastapi.security import OAuth2PasswordBearer 
from schemas import UserData

SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-fallback-secret-key-for-local-dev")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24


pwd_context = CryptContext(schemas=['bcrypt'] , deprecated='auto')
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/login")

FAKE_USERS_DB = {
    "researcher": {
        "username": "researcher",
        "hashed_password": pwd_context.hash("password123"),
        "tenant_id": "org_default"
    }
}


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password , hashed_password)


def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode , SECRET_KEY , algorithm=ALGORITHM)

def get_current_user(token:str= Depends(oauth2_scheme)) -> UserData:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY , algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        tenant_id: str = payload.get("tenant_id")
        if username is None or username not in FAKE_USERS_DB:
            raise credentials_exception
        return UserData(username=username , tenant_id=tenant_id)
    except JWTError:
        raise credentials_exception