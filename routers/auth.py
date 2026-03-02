from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import timedelta
from dependencies import (
    fake_users_db, revoked_tokens, verify_password,
    create_access_token, get_current_user,
    require_role, ACCESS_TOKEN_EXPIRE_MINUTES
)

router = APIRouter()

class RoleUpdate(BaseModel):
    new_role: str  # "clinician" | "admin" | "manager" | "it_security"

# ── POST /auth/login ─────────────────────────────────────────────────────────
@router.post("/login", summary="Login and get JWT token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Authenticate with username + password.
    Returns a JWT bearer token valid for 60 minutes.

    **Sample credentials:**
    - username: `dr.ahmad` / password: `password123`
    - username: `admin` / password: `admin123`
    """
    user = fake_users_db.get(form_data.username)
    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    token = create_access_token(
        data={"sub": user["username"]},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": f"{ACCESS_TOKEN_EXPIRE_MINUTES} minutes",
        "user": {
            "user_id": user["user_id"],
            "username": user["username"],
            "full_name": user["full_name"],
            "role": user["role"],
        }
    }

# ── POST /auth/logout ────────────────────────────────────────────────────────
@router.post("/logout", summary="Revoke current token")
def logout(token: str = Depends(__import__("dependencies").oauth2_scheme)):
    """
    Blacklists the current JWT token so it can no longer be used.
    """
    revoked_tokens.add(token)
    return {"message": "Successfully logged out. Token revoked."}

# ── GET /auth/me ─────────────────────────────────────────────────────────────
@router.get("/me", summary="Get current logged-in user")
def get_me(current_user=Depends(get_current_user)):
    """
    Returns the profile of the currently authenticated user.

    **Sample Response:**
    ```json
    {
      "user_id": "u001",
      "username": "dr.ahmad",
      "full_name": "Dr. Ahmad Khalil",
      "role": "clinician"
    }
    ```
    """
    return {
        "user_id": current_user["user_id"],
        "username": current_user["username"],
        "full_name": current_user["full_name"],
        "role": current_user["role"],
    }

# ── PUT /users/{user_id}/role ─────────────────────────────────────────────────
@router.put("/users/{user_id}/role", summary="Update a user's role (admin only)")
def update_role(
    user_id: str,
    body: RoleUpdate,
    current_user=Depends(require_role("admin"))
):
    """
    Change the role of a user. **Admin only.**

    Valid roles: `clinician`, `admin`, `manager`, `it_security`
    """
    valid_roles = {"clinician", "admin", "manager", "it_security"}
    if body.new_role not in valid_roles:
        raise HTTPException(status_code=400, detail=f"Invalid role. Choose from: {valid_roles}")

    for uname, user in fake_users_db.items():
        if user["user_id"] == user_id:
            old_role = user["role"]
            user["role"] = body.new_role
            return {
                "message": "Role updated successfully",
                "user_id": user_id,
                "old_role": old_role,
                "new_role": body.new_role,
                "updated_by": current_user["username"],
            }

    raise HTTPException(status_code=404, detail="User not found")
