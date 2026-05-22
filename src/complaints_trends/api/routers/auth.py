from __future__ import annotations

from fastapi import APIRouter, Depends, Header, HTTPException

from ..deps import get_service_container
from ..schemas import (
    AuthLoginRequest,
    AuthPasswordChangeRequest,
    AuthProfileUpdateRequest,
    AuthRegisterRequest,
    AuthTokenResponse,
    AuthUser,
)

router = APIRouter(prefix="/api/auth", tags=["auth"])


def _public_user(user: dict) -> AuthUser:
    return AuthUser(
        id=str(user["id"]),
        email=str(user["email"]),
        display_name=str(user["display_name"]),
        first_name=str(user.get("first_name") or ""),
        last_name=str(user.get("last_name") or ""),
        role=str(user.get("role") or "user"),
        workspace_id=str(user.get("workspace_id") or "default"),
        workspace_role=str(user.get("workspace_role") or user.get("role") or "user"),
    )


@router.post("/register", response_model=AuthTokenResponse)
def register(req: AuthRegisterRequest, services=Depends(get_service_container)):
    try:
        user = services["catalog"].create_user(req.email, req.password, req.display_name)
        token = services["catalog"].issue_access_token(str(user["id"]))
        return AuthTokenResponse(**token, user=_public_user(user))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/login", response_model=AuthTokenResponse)
def login(req: AuthLoginRequest, services=Depends(get_service_container)):
    user = services["catalog"].authenticate_user(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    token = services["catalog"].issue_access_token(str(user["id"]))
    return AuthTokenResponse(**token, user=_public_user(user))


@router.get("/me", response_model=AuthUser)
def me(authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    return _public_user(user)


@router.patch("/me", response_model=AuthUser)
def update_me(req: AuthProfileUpdateRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        updated = services["catalog"].update_profile(
            str(user["id"]),
            first_name=req.first_name,
            last_name=req.last_name,
            display_name=req.display_name,
        )
        return _public_user(updated)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/me/password")
def change_password(req: AuthPasswordChangeRequest, authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    user = services["catalog"].user_from_authorization(authorization)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required.")
    try:
        services["catalog"].change_password(str(user["id"]), req.current_password, req.new_password)
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/logout")
def logout(authorization: str | None = Header(default=None), services=Depends(get_service_container)):
    services["catalog"].revoke_authorization(authorization)
    return {"ok": True}
