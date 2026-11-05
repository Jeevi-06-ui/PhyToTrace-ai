"""PhytoTrace AI — MongoDB Atlas & SQLite Resilient Persistence Layer.

Supports MongoDB Atlas collections with automatic fallback to local SQLite
when MONGODB_URI is absent or unreachable in development.
In production mode, live MongoDB Atlas connection is strictly enforced.
"""

from datetime import datetime, timezone
import os
from typing import Any, Dict, List, Optional
from phytotrace.db.database import InspectionRepository
from phytotrace.utils.logging import get_logger

logger = get_logger("phytotrace.db.mongo")


class PhytoTraceDatabase:
    """Unified repository managing users, farms, plants, inspections, evidence regions,
    weather snapshots, assistant sessions, and user preferences on MongoDB Atlas.
    """

    def __init__(self, uri: Optional[str] = None, db_name: str = "phytotrace"):
        self.uri = os.getenv("MONGODB_URI", "") if uri is None else uri
        self.db_name = os.getenv("MONGODB_DB_NAME", db_name)
        self.mongo_client = None
        self.db = None
        self.is_mongo = False

        production = os.getenv("PHYTOTRACE_ENV") == "production"
        self.sqlite_repo = None
        if not production and (os.getenv("PHYTOTRACE_ALLOW_LOCAL_SQLITE") == "1" or uri == ""):
            self.sqlite_repo = InspectionRepository(os.getenv("PHYTOTRACE_SQLITE_PATH", "data/phytotrace.db"))
        self._local_assistant_sessions: Dict[str, List[Dict[str, Any]]] = {}
        self._local_preferences: Dict[str, Dict[str, Any]] = {}

        if self.uri and os.getenv("PHYTOTRACE_ROTATED_CLOUD_CREDENTIALS", "1") == "1":
            try:
                from pymongo import MongoClient
                self.mongo_client = MongoClient(
                    self.uri,
                    serverSelectionTimeoutMS=4000,
                    connectTimeoutMS=4000,
                )
                # Verify connection with ping
                self.mongo_client.admin.command("ping")
                self.db = self.mongo_client[self.db_name]
                self.is_mongo = True
                self._ensure_indexes()
                logger.info("Connected to MongoDB Atlas cluster (%s).", self.db_name)
            except Exception as e:
                logger.warning("MongoDB Atlas connection failed (%s)", type(e).__name__)
                if os.getenv("PHYTOTRACE_ENV") == "production":
                    raise RuntimeError("Production mode requires a live MongoDB Atlas connection.") from e
                self.is_mongo = False
        else:
            if os.getenv("PHYTOTRACE_ENV") == "production":
                raise RuntimeError(
                    "Production requires MongoDB Atlas with rotated credentials. "
                    "Silent SQLite fallback is disabled in production."
                )
            logger.info("MongoDB Atlas is disabled. Local SQLite enabled: %s.", self.sqlite_repo is not None)

    def _ensure_indexes(self) -> None:
        """Create query indexes on all standard MongoDB collections."""
        if not self.is_mongo or self.db is None:
            return
        try:
            self.db.users.create_index("firebase_uid", unique=True)
            self.db.farms.create_index([("owner_uid", 1), ("farm_id", 1)])
            self.db.plants.create_index([("farm_id", 1), ("plant_id", 1)])
            self.db.inspections.create_index([("owner_uid", 1), ("created_at", -1)])
            self.db.inspections.create_index("id", unique=True)
            self.db.inspections.create_index([("plant_id", 1), ("created_at", -1)])
            self.db.evidence_regions.create_index([("inspection_id", 1), ("label", 1)])
            self.db.weather_snapshots.create_index([("location", 1), ("timestamp", -1)])
            self.db.assistant_sessions.create_index([("owner_uid", 1), ("session_id", 1)])
            self.db.user_preferences.create_index("firebase_uid", unique=True)
        except Exception as e:
            logger.warning("Could not create indexes on MongoDB: %s", e)

    # -------------------------------------------------------------------------
    # Users & Preferences
    # -------------------------------------------------------------------------
    def upsert_user(
        self,
        firebase_uid: str,
        email: str = "",
        display_name: str = "",
        preferred_language: str = "en",
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        user_doc = {
            "firebase_uid": firebase_uid,
            "email": email,
            "display_name": display_name or email.split("@")[0] or "Agronomist",
            "preferred_language": preferred_language,
            "updated_at": now,
        }
        if self.is_mongo and self.db is not None:
            try:
                self.db.users.update_one(
                    {"firebase_uid": firebase_uid},
                    {"$set": user_doc, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
            except Exception as e:
                logger.error("Mongo upsert_user error: %s", e)

        return user_doc

    def get_user(self, firebase_uid: str) -> Optional[Dict[str, Any]]:
        if self.is_mongo and self.db is not None:
            try:
                doc = self.db.users.find_one({"firebase_uid": firebase_uid}, {"_id": 0})
                if doc:
                    return doc
            except Exception as e:
                logger.error("Mongo get_user error: %s", e)
        return {
            "firebase_uid": firebase_uid,
            "email": "",
            "display_name": "Field Agronomist",
            "preferred_language": "en",
        }

    def save_user_preferences(self, firebase_uid: str, preferences: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        pref_doc = {
            "firebase_uid": firebase_uid,
            "preferences": preferences,
            "updated_at": now,
        }
        if self.is_mongo and self.db is not None:
            try:
                self.db.user_preferences.update_one(
                    {"firebase_uid": firebase_uid},
                    {"$set": pref_doc, "$setOnInsert": {"created_at": now}},
                    upsert=True,
                )
            except Exception as e:
                logger.error("Mongo save_user_preferences error: %s", e)

        self._local_preferences[firebase_uid] = pref_doc
        return pref_doc

    def save_weather_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """Store weather observation snapshot with timestamp."""
        if self.is_mongo and self.db is not None:
            try:
                self.db.weather_snapshots.insert_one({
                    **snapshot,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                })
            except Exception as e:
                logger.warning("Could not save weather snapshot: %s", e)

    def get_user_preferences(self, firebase_uid: str) -> Dict[str, Any]:
        if self.is_mongo and self.db is not None:
            try:
                doc = self.db.user_preferences.find_one({"firebase_uid": firebase_uid}, {"_id": 0})
                if doc and "preferences" in doc:
                    return doc["preferences"]
            except Exception as e:
                logger.error("Mongo get_user_preferences error: %s", e)
        return self._local_preferences.get(firebase_uid, {}).get("preferences", {})

    # -------------------------------------------------------------------------
    # Inspections (Enforcing User Ownership & Data Isolation)
    # -------------------------------------------------------------------------
    def save_inspection(self, inspection_data: Dict[str, Any], owner_uid: str = "guest") -> str:
        insp_id = inspection_data.get("id") or inspection_data.get("inspection_id")
        if not insp_id:
            raise ValueError("Inspection record must contain an 'id'.")

        data_copy = dict(inspection_data)
        data_copy["owner_uid"] = owner_uid
        if "created_at" not in data_copy:
            data_copy["created_at"] = datetime.now(timezone.utc).isoformat()

        if self.is_mongo:
            # Embed evidence in the same document so the inspection is saved atomically.
            clean_doc = {key: value for key, value in data_copy.items() if not key.startswith("_")}
            self.db.inspections.insert_one(clean_doc)
        elif self.sqlite_repo is not None:
            self.sqlite_repo.create_inspection(data_copy)
        else:
            raise RuntimeError("MongoDB Atlas is required for account-owned inspections.")

        return insp_id

    def get_inspection(self, insp_id: str, owner_uid: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Retrieve inspection enforcing owner isolation."""
        if self.is_mongo and self.db is not None:
            try:
                query: Dict[str, Any] = {"id": insp_id}
                if owner_uid:
                    query["owner_uid"] = owner_uid
                doc = self.db.inspections.find_one(query, {"_id": 0})
                return doc
            except Exception as e:
                logger.error("MongoDB get_inspection error: %s", e)
                if os.getenv("PHYTOTRACE_ENV") == "production":
                    raise

        if self.sqlite_repo is not None and owner_uid is None:
            return self.sqlite_repo.get_inspection(insp_id)
        return None

    def list_inspections(
        self,
        farm_id: Optional[str] = None,
        plant_id: Optional[str] = None,
        severity_tier: Optional[str] = None,
        owner_uid: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """List inspections strictly isolated to the requesting user/guest."""
        if self.is_mongo and self.db is not None:
            try:
                query: Dict[str, Any] = {}
                if farm_id:
                    query["farm_id"] = farm_id
                if plant_id:
                    query["plant_id"] = plant_id
                if severity_tier:
                    query["severity_tier"] = severity_tier
                if owner_uid:
                    query["owner_uid"] = owner_uid

                total = self.db.inspections.count_documents(query)
                cursor = (
                    self.db.inspections.find(query, {"_id": 0})
                    .sort("created_at", -1)
                    .skip(offset)
                    .limit(limit)
                )
                items = list(cursor)
                return {
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "items": items,
                }
            except Exception as e:
                logger.warning("MongoDB inspection list failed (%s)", type(e).__name__)
                raise

        if owner_uid is not None or self.sqlite_repo is None:
            return {"total": 0, "limit": limit, "offset": offset, "items": []}
        return self.sqlite_repo.list_inspections(
            farm_id=farm_id,
            plant_id=plant_id,
            severity_tier=severity_tier,
            limit=limit,
            offset=offset,
        )

    # -------------------------------------------------------------------------
    # Farms & Plants
    # -------------------------------------------------------------------------
    def get_farms_overview(self, owner_uid: Optional[str] = None) -> Dict[str, Any]:
        return self.sqlite_repo.get_farm_overview()

    def get_plant_timeline(self, plant_id: str, owner_uid: Optional[str] = None) -> Dict[str, Any]:
        return self.sqlite_repo.get_plant_timeline(plant_id)

    # -------------------------------------------------------------------------
    # Assistant Chat Sessions
    # -------------------------------------------------------------------------
    def save_assistant_message(self, session_id: str, message: Dict[str, Any], owner_uid: str) -> None:
        if self.is_mongo:
            now = datetime.now(timezone.utc).isoformat()
            session_key = {"owner_uid": owner_uid, "session_id": session_id}
            self.db.assistant_sessions.update_one(
                session_key,
                {"$push": {"messages": {"$each": [{**message, "timestamp": now}], "$slice": -20}},
                 "$set": {"updated_at": now}, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )

    def get_assistant_messages(self, session_id: str, owner_uid: str) -> List[Dict[str, Any]]:
        if self.is_mongo:
            doc = self.db.assistant_sessions.find_one({"owner_uid": owner_uid, "session_id": session_id}, {"messages": 1})
            return doc.get("messages", []) if doc else []
        return []


# Global singleton instance
_db_instance: Optional[PhytoTraceDatabase] = None


def get_db() -> PhytoTraceDatabase:
    global _db_instance
    if _db_instance is None:
        _db_instance = PhytoTraceDatabase()
    return _db_instance
