"""PhytoTrace AI — SQLite / PostgreSQL-compatible database and inspection repository.

Stores and queries inspections, plant monitoring timelines, and farm metrics.
"""

from datetime import datetime
import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from phytotrace.utils.logging import get_logger
from phytotrace.utils.paths import resolve_path

logger = get_logger("phytotrace.db")

DEFAULT_DB_PATH = "data/phytotrace.db"


class InspectionRepository:
    """Repository managing inspection records, longitudinal progression, and farm stats."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = resolve_path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS inspections (
                id TEXT PRIMARY KEY,
                farm_id TEXT NOT NULL,
                plant_id TEXT NOT NULL,
                leaf_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                original_image_path TEXT NOT NULL,
                annotated_image_path TEXT,
                leaf_mask_path TEXT,
                mine_mask_path TEXT,
                quality_status TEXT NOT NULL,
                leaf_area_pixels INTEGER,
                damage_detected INTEGER,
                damage_area_pixels INTEGER,
                damage_percentage REAL,
                severity_tier TEXT NOT NULL,
                decision TEXT NOT NULL,
                uncertainty_status TEXT,
                explanation TEXT,
                safety_notice TEXT,
                model_version TEXT NOT NULL,
                processing_time_ms INTEGER,
                regions_json TEXT,
                notes TEXT
            );
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_farm ON inspections(farm_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_plant ON inspections(plant_id);
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_inspections_created ON inspections(created_at DESC);
            """)
            conn.commit()
            logger.info("Database initialized at %s", self.db_path)

    def create_inspection(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert a new inspection record into the database."""
        insp_id = data.get("id") or f"INSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{data.get('plant_id', 'PL001')}"
        created_at = data.get("created_at") or datetime.now().isoformat()
        regions_json = json.dumps(data.get("damage_regions", [])) if isinstance(data.get("damage_regions"), list) else str(data.get("regions_json", "[]"))

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT OR REPLACE INTO inspections (
                id, farm_id, plant_id, leaf_id, created_at, original_filename,
                original_image_path, annotated_image_path, leaf_mask_path, mine_mask_path,
                quality_status, leaf_area_pixels, damage_detected, damage_area_pixels,
                damage_percentage, severity_tier, decision, uncertainty_status,
                explanation, safety_notice, model_version, processing_time_ms,
                regions_json, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insp_id,
                data.get("farm_id", "FARM_MAIN"),
                data.get("plant_id", "PLANT_001"),
                data.get("leaf_id", "LEAF_01"),
                created_at,
                data.get("original_filename", "unnamed.jpg"),
                str(data.get("original_image_path", "")),
                str(data.get("annotated_image_path", "")),
                str(data.get("leaf_mask_path", "")),
                str(data.get("mine_mask_path", "")),
                data.get("quality_status", "acceptable"),
                data.get("leaf_area_pixels"),
                data.get("damage_detected"),
                data.get("damage_area_pixels"),
                data.get("damage_percentage"),
                data.get("severity_tier", "none"),
                data.get("decision", "no_visible_mine"),
                data.get("uncertainty_status", "normal"),
                data.get("explanation", ""),
                data.get("safety_notice", ""),
                data.get("model_version", "PhytoTrace-v0.1.0"),
                data.get("processing_time_ms", 0),
                regions_json,
                data.get("notes", ""),
            ))
            conn.commit()

        return self.get_inspection(insp_id)

    # Alias for API uniformity
    save_inspection = create_inspection

    def get_inspection(self, insp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single inspection by its ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM inspections WHERE id = ?", (insp_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_dict(row)

    def list_inspections(
        self,
        farm_id: Optional[str] = None,
        plant_id: Optional[str] = None,
        severity_tier: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """List inspections with optional filters, ordered chronologically newest first."""
        query = "SELECT * FROM inspections WHERE 1=1"
        params: List[Any] = []

        if farm_id:
            query += " AND farm_id = ?"
            params.append(farm_id)
        if plant_id:
            query += " AND plant_id = ?"
            params.append(plant_id)
        if severity_tier:
            query += " AND severity_tier = ?"
            params.append(severity_tier)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [self._row_to_dict(r) for r in cursor.fetchall()]

    def count_inspections(
        self,
        farm_id: Optional[str] = None,
        plant_id: Optional[str] = None,
        severity_tier: Optional[str] = None,
    ) -> int:
        """Count total matching inspections."""
        query = "SELECT COUNT(*) as cnt FROM inspections WHERE 1=1"
        params: List[Any] = []

        if farm_id:
            query += " AND farm_id = ?"
            params.append(farm_id)
        if plant_id:
            query += " AND plant_id = ?"
            params.append(plant_id)
        if severity_tier:
            query += " AND severity_tier = ?"
            params.append(severity_tier)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return int(row["cnt"]) if row else 0

    def get_plant_timeline(self, plant_id: str) -> List[Dict[str, Any]]:
        """Retrieve longitudinal history for a specific plant with progression deltas."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM inspections WHERE plant_id = ? ORDER BY created_at ASC",
                (plant_id,),
            )
            records = [self._row_to_dict(r) for r in cursor.fetchall()]

        # Calculate progression deltas compared to preceding inspection
        for idx, rec in enumerate(records):
            if idx == 0:
                rec["progression_delta"] = 0.0
                rec["progression_status"] = "baseline_established"
            else:
                prev_dmg = records[idx - 1].get("damage_percentage")
                curr_dmg = rec.get("damage_percentage")
                delta = round(curr_dmg - prev_dmg, 2) if curr_dmg is not None and prev_dmg is not None else None
                rec["progression_delta"] = delta
                if delta is None:
                    rec["progression_status"] = "measurement_unavailable"
                elif delta > 1.0:
                    rec["progression_status"] = "infestation_expanding"
                elif delta < -1.0:
                    rec["progression_status"] = "infestation_receding"
                else:
                    rec["progression_status"] = "infestation_stable"

        return records

    def get_farm_overview(self) -> Dict[str, Any]:
        """Aggregate statistics across all farms and plants."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT 
                COUNT(*) as total_inspections,
                COUNT(DISTINCT farm_id) as total_farms,
                COUNT(DISTINCT plant_id) as total_plants,
                AVG(damage_percentage) as avg_damage_pct,
                SUM(CASE WHEN damage_detected = 1 THEN 1 ELSE 0 END) as active_infestations
            FROM inspections;
            """)
            summary_row = cursor.fetchone()

            cursor.execute("""
            SELECT severity_tier, COUNT(*) as cnt
            FROM inspections
            GROUP BY severity_tier;
            """)
            tier_rows = cursor.fetchall()
            tiers = {r["severity_tier"]: r["cnt"] for r in tier_rows}

            cursor.execute("""
            SELECT farm_id, COUNT(*) as insp_count, COUNT(DISTINCT plant_id) as plant_count,
                   AVG(damage_percentage) as avg_dmg
            FROM inspections
            GROUP BY farm_id;
            """)
            farms = [
                {
                    "farm_id": r["farm_id"],
                    "inspections": r["insp_count"],
                    "plants": r["plant_count"],
                    "average_damage_pct": round(float(r["avg_dmg"] or 0.0), 2),
                }
                for r in cursor.fetchall()
            ]

        return {
            "total_inspections": summary_row["total_inspections"] if summary_row else 0,
            "total_farms": summary_row["total_farms"] if summary_row else 0,
            "total_plants": summary_row["total_plants"] if summary_row else 0,
            "average_damage_pct": round(float(summary_row["avg_damage_pct"] or 0.0), 2) if summary_row else 0.0,
            "active_infestations": summary_row["active_infestations"] if summary_row else 0,
            "severity_distribution": tiers,
            "farms": farms,
        }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        if d["damage_detected"] is not None:
            d["damage_detected"] = bool(d["damage_detected"])
        if d.get("regions_json"):
            try:
                d["damage_regions"] = json.loads(d["regions_json"])
            except Exception:
                d["damage_regions"] = []
        else:
            d["damage_regions"] = []
        return d

