# backend/create_admin.py

from core.database import SessionLocal
from core.models import User
from services.security import get_password_hash


def main():
    db = SessionLocal()

    try:
        email = "anes@example.com"

        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            existing_user.first_name = "anes"
            existing_user.last_name = "ferdjani"
            existing_user.hashed_password = get_password_hash("password123")
            existing_user.role = "admin"

            db.commit()
            db.refresh(existing_user)

            print("✅ Existing user updated to admin.")
            print(f"Email: {email}")
            print("Password: password123")
            return

        admin = User(
            first_name="anes",
            last_name="ferdjani",
            email=email,
            hashed_password=get_password_hash("password123"),
            role="admin",
        )

        db.add(admin)
        db.commit()
        db.refresh(admin)

        print("✅ Admin user created.")
        print(f"ID: {admin.id}")
        print(f"Email: {email}")
        print("Password: password123")

    except Exception as e:
        db.rollback()
        print(f"❌ Error creating admin: {e}")

    finally:
        db.close()


if __name__ == "__main__":
    main()