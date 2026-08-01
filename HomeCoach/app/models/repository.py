import json
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta
from pathlib import Path


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


class CoachingRepository:
    def __init__(self, database_path):
        self.database_path = str(database_path)

    def _connect(self):
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def init_schema(self):
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS coaching_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    child_name TEXT NOT NULL,
                    material TEXT NOT NULL,
                    material_id TEXT NOT NULL DEFAULT '176',
                    status TEXT NOT NULL DEFAULT 'active',
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    duration_seconds INTEGER NOT NULL DEFAULT 0,
                    average_wait REAL NOT NULL DEFAULT 0,
                    expansion_rate INTEGER NOT NULL DEFAULT 0,
                    turn_taking_rate INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS dialogue_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    text TEXT NOT NULL,
                    pause_before REAL NOT NULL DEFAULT 0,
                    gaze_on_target INTEGER NOT NULL DEFAULT 1,
                    analysis_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id)
                        REFERENCES coaching_sessions(id)
                        ON DELETE CASCADE
                );
                """
            )
            session_columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(coaching_sessions)"
                ).fetchall()
            }
            if "material_id" not in session_columns:
                connection.execute(
                    """
                    ALTER TABLE coaching_sessions
                    ADD COLUMN material_id TEXT NOT NULL DEFAULT '176'
                    """
                )
            connection.commit()

    def create_session(
        self,
        child_name,
        material,
        material_id="176",
        started_at=None,
    ):
        started_at = started_at or _now_iso()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO coaching_sessions
                    (child_name, material, material_id, status, started_at)
                VALUES (?, ?, ?, 'active', ?)
                """,
                (child_name, material, material_id, started_at),
            )
            connection.commit()
            return self.get_session(cursor.lastrowid)

    def get_session(self, session_id):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM coaching_sessions WHERE id = ?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_sessions(self, limit=20):
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    s.*,
                    COUNT(e.id) AS event_count
                FROM coaching_sessions AS s
                LEFT JOIN dialogue_events AS e ON e.session_id = s.id
                GROUP BY s.id
                ORDER BY s.started_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [dict(row) for row in rows]

    def add_event(
        self,
        session_id,
        speaker,
        text,
        pause_before,
        gaze_on_target,
        analysis,
        created_at=None,
    ):
        created_at = created_at or _now_iso()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                INSERT INTO dialogue_events (
                    session_id,
                    speaker,
                    text,
                    pause_before,
                    gaze_on_target,
                    analysis_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    speaker,
                    text,
                    pause_before,
                    int(gaze_on_target),
                    json.dumps(analysis, ensure_ascii=False),
                    created_at,
                ),
            )
            connection.commit()
            return self.get_event(cursor.lastrowid)

    def get_event(self, event_id):
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM dialogue_events WHERE id = ?",
                (event_id,),
            ).fetchone()
            return self._event_to_dict(row) if row else None

    def update_event_analysis(self, event_id, analysis):
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE dialogue_events
                SET analysis_json = ?
                WHERE id = ?
                """,
                (json.dumps(analysis, ensure_ascii=False), event_id),
            )
            connection.commit()
            return self.get_event(event_id) if cursor.rowcount else None

    def list_events(self, session_id):
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT * FROM dialogue_events
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [self._event_to_dict(row) for row in rows]

    def update_metrics(self, session_id, metrics):
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE coaching_sessions
                SET average_wait = ?,
                    expansion_rate = ?,
                    turn_taking_rate = ?
                WHERE id = ?
                """,
                (
                    metrics["average_wait"],
                    metrics["expansion_rate"],
                    metrics["turn_taking_rate"],
                    session_id,
                ),
            )
            connection.commit()

    def finish_session(self, session_id, ended_at=None):
        session = self.get_session(session_id)
        if not session:
            return None

        ended_at = ended_at or _now_iso()
        try:
            start = datetime.fromisoformat(session["started_at"])
            end = datetime.fromisoformat(ended_at)
            duration = max(1, int((end - start).total_seconds()))
        except (TypeError, ValueError):
            duration = 0

        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE coaching_sessions
                SET status = 'completed',
                    ended_at = ?,
                    duration_seconds = ?
                WHERE id = ?
                """,
                (ended_at, duration, session_id),
            )
            connection.commit()
        return self.get_session(session_id)

    def get_summary(self):
        week_start = (datetime.now().astimezone() - timedelta(days=7)).isoformat(
            timespec="seconds"
        )
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    COUNT(*) AS weekly_sessions,
                    COALESCE(ROUND(AVG(average_wait), 1), 0) AS average_wait,
                    COALESCE(ROUND(AVG(expansion_rate)), 0) AS expansion_rate,
                    COALESCE(ROUND(AVG(turn_taking_rate)), 0) AS turn_taking_rate,
                    COALESCE(SUM(duration_seconds), 0) AS total_seconds
                FROM coaching_sessions
                WHERE started_at >= ?
                """,
                (week_start,),
            ).fetchone()
            return dict(row)

    def seed_demo_data(self):
        with closing(self._connect()) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM coaching_sessions"
            ).fetchone()[0]
        if count:
            return

        started = datetime.now().astimezone() - timedelta(days=1, minutes=12)
        session = self.create_session(
            "小宇",
            "玩具描述練習",
            material_id="162",
            started_at=started.isoformat(timespec="seconds"),
        )
        seed_events = [
            (
                "parent",
                "你看到什麼？",
                0.0,
                True,
                {"wait_met": None, "expansion_met": None, "turn_taking": True},
            ),
            (
                "child",
                "球球",
                3.8,
                True,
                {"wait_met": None, "expansion_met": None, "turn_taking": True},
            ),
            (
                "parent",
                "對，是紅色的球球",
                3.4,
                True,
                {"wait_met": True, "expansion_met": True, "turn_taking": True},
            ),
            (
                "child",
                "紅色球球",
                2.7,
                True,
                {"wait_met": None, "expansion_met": None, "turn_taking": True},
            ),
            (
                "parent",
                "紅色球球滾過來了",
                3.2,
                True,
                {"wait_met": True, "expansion_met": True, "turn_taking": True},
            ),
            (
                "child",
                "滾過來",
                2.9,
                False,
                {"wait_met": None, "expansion_met": None, "turn_taking": True},
            ),
        ]
        for speaker, text, pause, gaze, flags in seed_events:
            analysis = {
                **flags,
                "gaze_available": True,
                "gaze_on_target": gaze,
                "suggestion": {
                    "tone": "positive",
                    "eyebrow": "做得很好",
                    "title": "接住孩子的話",
                    "message": "你有先等待，再把孩子的詞延伸成完整語句。",
                    "example": "「對，是紅色的球球。」",
                },
            }
            self.add_event(
                session["id"],
                speaker,
                text,
                pause,
                gaze,
                analysis,
                created_at=(started + timedelta(seconds=18)).isoformat(
                    timespec="seconds"
                ),
            )

        self.update_metrics(
            session["id"],
            {
                "average_wait": 3.3,
                "expansion_rate": 100,
                "turn_taking_rate": 100,
            },
        )
        self.finish_session(
            session["id"],
            ended_at=(started + timedelta(minutes=8, seconds=42)).isoformat(
                timespec="seconds"
            ),
        )

    @staticmethod
    def _event_to_dict(row):
        event = dict(row)
        event["gaze_on_target"] = bool(event["gaze_on_target"])
        event["analysis"] = json.loads(event.pop("analysis_json"))
        event["gaze_available"] = bool(
            event["analysis"].get("gaze_available", True)
        )
        return event
