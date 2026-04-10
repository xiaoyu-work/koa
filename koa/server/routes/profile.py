"""Profile extraction routes (internal, service-to-service)."""

import logging
from typing import Optional

from fastapi import APIRouter, Request

from ...errors import E, KoaError
from ...providers.email.factory import EmailProviderFactory
from ...providers.email.resolver import AccountResolver
from ...services.profile_extraction import ProfileExtractionService
from ...services.profile_repo import ProfileRepository
from ..app import require_app, verify_service_key

logger = logging.getLogger(__name__)
router = APIRouter()

# Singleton service instance (in-memory job tracking)
_service = ProfileExtractionService()


@router.post("/api/internal/profile/extract")
async def start_profile_extraction(
    request: Request,
    tenant_id: str,
    email_account: Optional[str] = None,
    callback_url: Optional[str] = None,
):
    """
    Start profile extraction for a tenant.

    If email_account is provided, only scan that account and LLM-merge
    with the existing profile. Otherwise scan all linked email accounts.

    If callback_url is provided, POST the extracted profile there on completion.

    Requires X-Service-Key header.
    Returns a job_id for status polling.
    """
    verify_service_key(request)
    app = require_app()
    await app._ensure_initialized()

    # Resolve email accounts
    try:
        if email_account:
            accounts = await AccountResolver.resolve_accounts(tenant_id, [email_account])
        else:
            accounts = await AccountResolver.resolve_accounts(tenant_id, ["all"])
    except Exception as e:
        logger.error(f"Failed to resolve email accounts for {tenant_id}: {e}")
        raise KoaError(E.INTERNAL_ERROR, f"Failed to resolve email accounts: {e}")

    if not accounts:
        raise KoaError(
            E.NOT_FOUND,
            "No email accounts found for this tenant",
            details={"resource": "email_account"},
        )

    # Create email providers
    providers = []
    for acc in accounts:
        provider = EmailProviderFactory.create_provider(acc)
        if provider:
            providers.append(provider)

    if not providers:
        raise KoaError(E.INTERNAL_ERROR, "Failed to create email providers")

    # Forward service key so callback can authenticate
    callback_headers = {}
    service_key = request.headers.get("X-Service-Key")
    if service_key:
        callback_headers["X-Service-Key"] = service_key

    # Start background extraction (with DB persistence + callback)
    profile_repo = ProfileRepository(app._database)
    job_id = _service.start_extraction(
        tenant_id=tenant_id,
        providers=providers,
        llm_client=app._llm_client,
        profile_repo=profile_repo,
        callback_url=callback_url or "",
        callback_headers=callback_headers if callback_url else None,
        database=app._database,
    )

    logger.info(
        f"Profile extraction started: job={job_id}, tenant={tenant_id}, "
        f"email_account={email_account}, providers={len(providers)}, "
        f"callback={'yes' if callback_url else 'no'}"
    )
    return {"job_id": job_id, "status": "started"}


@router.get("/api/internal/profile/extract/{job_id}/status")
async def get_extraction_status(request: Request, job_id: str):
    """
    Get the status of a profile extraction job.

    Requires X-Service-Key header.
    When status is "completed", the response includes the extracted profile.
    """
    verify_service_key(request)

    job = _service.get_job_status(job_id)
    if not job:
        raise KoaError(
            E.NOT_FOUND, "Extraction job not found", details={"resource": "extraction_job"}
        )

    result = {
        "job_id": job["job_id"],
        "status": job["status"],
        "progress": job.get("progress", {}),
    }

    if job["status"] == "completed" and job.get("profile"):
        result["profile"] = job["profile"]

    if job["status"] == "failed":
        result["error"] = job.get("error")

    return result


@router.get("/api/internal/profile/{tenant_id}")
async def get_tenant_profile(request: Request, tenant_id: str):
    """Get the stored profile for a tenant. Returns 404 if not extracted yet."""
    verify_service_key(request)
    app = require_app()
    await app._ensure_initialized()

    repo = ProfileRepository(app._database)
    profile = await repo.get_profile(tenant_id)
    if profile is None:
        raise KoaError(
            E.NOT_FOUND, "No profile found for this tenant", details={"resource": "profile"}
        )
    return {"tenant_id": tenant_id, "profile": profile}
