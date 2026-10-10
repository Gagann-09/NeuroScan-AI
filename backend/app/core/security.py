"""
Firebase Authentication security utilities.

Provides lazy Firebase Admin SDK initialization and ID token verification
with revocation checking for FastAPI dependency injection.
"""
from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, status, Depends
from firebase_admin import auth, initialize_app, credentials, _apps
from firebase_admin.exceptions import FirebaseError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import User


@lru_cache(maxsize=1)
def _get_firebase_app() -> None:
    """
    Lazily initialize the Firebase Admin SDK app.

    Uses Application Default Credentials (ADC) which respects
    GOOGLE_APPLICATION_CREDENTIALS environment variable.
    The project ID is validated against the configured FIREBASE_PROJECT_ID.
    """
    if _apps:
        return

    settings = get_settings()
    cred = credentials.ApplicationDefault()
    initialize_app(cred, {"projectId": settings.FIREBASE_PROJECT_ID})


async def verify_id_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verify a Firebase ID token from the Authorization header.

    Args:
        authorization: The Authorization header value (Bearer <token>)

    Returns:
        dict: Decoded token claims (uid, email, email_verified, etc.)

    Raises:
        HTTPException: 401 for missing, malformed, invalid, expired, or revoked tokens.
                       All failure modes return the same generic message.
    """
    _get_firebase_app()

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )

    token = authorization.split(" ")[1]

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )

    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        return decoded_token
    except FirebaseError:
        # Generic message for all token validation failures:
        # - ExpiredIdTokenError
        # - InvalidIdTokenError
        # - RevokedIdTokenError
        # - CertificateFetchError
        # - Project ID mismatch
        # - Any other FirebaseError
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )


async def get_current_user(
    token_claims: dict = Depends(verify_id_token),
    db: Session = Depends(get_db),
) -> User:
    """
    Get or create the User record for the authenticated Firebase user.
    
    This dependency should be used by all protected endpoints to get the
    current user's database record, ensuring ownership enforcement.
    """
    firebase_uid = token_claims.get("uid")
    email = token_claims.get("email")
    
    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    
    # Get or create user record
    user = db.query(User).filter(User.firebase_uid == firebase_uid).first()
    if not user:
        user = User(
            firebase_uid=firebase_uid,
            email=email,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    else:
        # Update last_seen_at
        from datetime import datetime, timezone
        user.last_seen_at = datetime.now(timezone.utc)
        db.commit()
    
    return user


# Backward-compatible alias for dependency injection
get_current_user_claims = verify_id_token