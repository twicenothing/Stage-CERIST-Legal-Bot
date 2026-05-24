from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session, joinedload
from typing import Optional
from pydantic import BaseModel
from core.database import get_db
from core.models import Report, Message, User
from services.security import get_admin_user

router = APIRouter(prefix="/admin/reports", tags=["Admin Reports"])

# ── Schémas Pydantic ──────────────────────────────────────────────────────────

class ReportStatusUpdate(BaseModel):
    status: str # Les valeurs attendues : "en_attente", "traite", "rejete"

# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/")
def get_all_reports(
    status: Optional[str] = None, 
    db: Session = Depends(get_db), 
    admin: User = Depends(get_admin_user)
):
    """
    Récupère la liste des signalements. 
    Permet de filtrer par statut (ex: /admin/reports?status=en_attente).
    """
    query = db.query(Report).options(
        joinedload(Report.message), # Charge le message incriminé pour éviter le problème N+1 requêtes
        joinedload(Report.user)     # Charge l'utilisateur qui a fait le signalement
    )

    if status:
        query = query.filter(Report.status == status)

    # Trier par date de création, du plus récent au plus ancien
    reports = query.order_by(Report.created_at.desc()).all()

    return [
        {
            "id": r.id,
            "reason": r.reason,
            "details": r.details,
            "status": r.status,
            "createdAt": r.created_at.isoformat(),
            "reporter": {
                "id": r.user.id if r.user else None,
                "email": r.user.email if r.user else "Anonyme",
            },
            "message": {
                "id": r.message.id if r.message else None,
                "role": r.message.role if r.message else None,
                # On renvoie les parts pour que l'admin puisse lire ce que l'IA a répondu
                "parts": r.message.parts if r.message else None 
            }
        }
        for r in reports
    ]

@router.patch("/{report_id}/status")
def update_report_status(
    report_id: str, 
    body: ReportStatusUpdate, 
    db: Session = Depends(get_db), 
    admin: User = Depends(get_admin_user)
):
    """
    Permet à un administrateur de changer le statut d'un signalement après vérification.
    """
    valid_statuses = ["en_attente", "traite", "rejete"]
    if body.status not in valid_statuses:
        raise HTTPException(status_code=400, detail=f"Statut invalide. Utilisez l'un de : {valid_statuses}")

    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Signalement introuvable")

    report.status = body.status
    db.commit()

    return {"status": "success", "message": "Le statut du signalement a été mis à jour", "new_status": report.status}

@router.delete("/{report_id}")
def delete_report(
    report_id: str, 
    db: Session = Depends(get_db), 
    admin: User = Depends(get_admin_user)
):
    """
    Supprime définitivement un signalement (utile pour faire le ménage dans la base).
    """
    report = db.query(Report).filter(Report.id == report_id).first()
    
    if not report:
        raise HTTPException(status_code=404, detail="Signalement introuvable")

    db.delete(report)
    db.commit()

    return {"status": "success", "message": "Signalement supprimé"}