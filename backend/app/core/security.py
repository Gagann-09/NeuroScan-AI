"""
Firebase Authentication security utilities.

Provides lazy Firebase Admin SDK initialization and ID token verification
with revocation checking for FastAPI dependency injection.
"""
import logging
from functools import lru_cache
from typing import Optional

from fastapi import Header, HTTPException, status, Depends
from firebase_admin import auth, initialize_app, credentials, _apps
from firebase_admin.exceptions import FirebaseError
from google.auth.exceptions import GoogleAuthError
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import User

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_firebase_app() -> None:
    """
    Lazily initialize the Firebase Admin SDK app.

    Uses Application Default Credentials (ADC) which respects
    GOOGLE_APPLICATION_CREDENTIALS environment variable.
    The project ID is validated against the configured FIREBASE_PROJECT_ID.

    Raises:
        HTTPException: 503 if Firebase initialization fails due to missing
                       credentials, project misconfiguration, or other
                       infrastructure errors. Details are logged for operators
                       but not exposed to clients.
    """
    if _apps:
        return

    settings = get_settings()
    try:
        cred = credentials.ApplicationDefault()
        initialize_app(cred, {"projectId": settings.FIREBASE_PROJECT_ID})
    except GoogleAuthError as e:
        # Missing ADC, invalid credentials, or other Google auth infrastructure failure
        logger.error("Firebase Admin SDK initialization failed: GoogleAuthError", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from e
    except Exception as e:
        # Project ID mismatch, configuration error, or other initialization failure
        logger.error("Firebase Admin SDK initialization failed: unexpected error", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from e


async def verify_id_token(authorization: Optional[str] = Header(None)) -> dict:
    """
    Verify a Firebase ID token from the Authorization header.

    Args:
        authorization: The Authorization header value (Bearer <token>)

    Returns:
        dict: Decoded token claims (uid, email, email_verified, etc.)

    Raises:
        HTTPException: 401 for missing, malformed, invalid, expired, or revoked tokens.
                       503 for Firebase infrastructure errors (network, service unavailable).
                       All failure modes return generic messages without internal details.
    """
    try:
        _get_firebase_app()
    except HTTPException:
        # _get_firebase_app already raises appropriate HTTPException
        raise

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
        # Token validation failures - fail closed with generic 401
        # - ExpiredIdTokenError
        # - InvalidIdTokenError
        # - RevokedIdTokenError
        # - CertificateFetchError
        # - Project ID mismatch (token issued for different project)
        # - Any other FirebaseError
        logger.warning("Firebase ID token validation failed", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    except GoogleAuthError as e:
        # Infrastructure failures during token verification (network, timeout, service unavailable)
        # These are server-side errors, not invalid credentials
        logger.error("Firebase ID token verification failed: GoogleAuthError", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from e


async def get_current_user(
    token_claims: dict = Depends(verify_id_token),
    db: Session = Depends(get_db),
) -> User:
    """
    Get or create the User record for the authenticated Firebase user.
    
    This dependency should be used by all protected endpoints to get the
    current user's database record, ensuring ownership enforcement.

    Raises:
        HTTPException: 401 if token claims lack uid.
                       503 if database operation fails.
    """
    firebase_uid = token_claims.get("uid")
    email = token_claims.get("email")
    
    if not firebase_uid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
        )
    
    # Get or create user record
    try:
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
    except SQLAlchemyError as e:
        # Database infrastructure failure - do not leak connection details
        logger.error("Database error during user lookup/creation", exc_info=True)
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service temporarily unavailable",
        ) from e
    
    return user


# Backward-compatible alias for dependency injection
get_current_user_claims = verify_id_token