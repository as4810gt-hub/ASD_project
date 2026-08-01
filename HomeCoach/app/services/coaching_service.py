import threading


class CoachingService:
    def __init__(
        self,
        repository,
        rule_engine,
        context_builder=None,
        coach_provider=None,
        asd_analysis_service=None,
    ):
        self.repository = repository
        self.rule_engine = rule_engine
        self.context_builder = context_builder
        self.coach_provider = coach_provider
        self.asd_analysis_service = asd_analysis_service
        self._session_locks = {}
        self._session_locks_guard = threading.Lock()

    def _session_lock(self, session_id):
        with self._session_locks_guard:
            return self._session_locks.setdefault(
                int(session_id),
                threading.Lock(),
            )

    def start_session(self, child_name, material, material_id="176"):
        return self.repository.create_session(
            child_name,
            material,
            material_id=material_id,
        )

    def get_session(self, session_id):
        session = self.repository.get_session(session_id)
        if not session:
            return None
        session["events"] = self.repository.list_events(session_id)
        return session

    def analyze_asd_signals(self, session_id, **signals):
        with self._session_lock(session_id):
            session = self.repository.get_session(session_id)
            if not session:
                raise LookupError("找不到這次互動紀錄")
            if session["status"] != "active":
                raise LookupError("這次互動已經結束")
            if not self.asd_analysis_service:
                raise LookupError("ASD 分析模組尚未設定")
            return self.asd_analysis_service.analyze(
                session_id=session_id,
                **signals,
            )

    def record_event(
        self,
        session_id,
        speaker,
        text,
        pause_before,
        gaze_on_target,
        gaze_available=True,
        metadata=None,
        defer_coach=False,
    ):
        with self._session_lock(session_id):
            return self._record_event_locked(
                session_id=session_id,
                speaker=speaker,
                text=text,
                pause_before=pause_before,
                gaze_on_target=gaze_on_target,
                gaze_available=gaze_available,
                metadata=metadata,
                defer_coach=defer_coach,
            )

    def _record_event_locked(
        self,
        session_id,
        speaker,
        text,
        pause_before,
        gaze_on_target,
        gaze_available,
        metadata,
        defer_coach,
    ):
        session = self.repository.get_session(session_id)
        if not session:
            raise LookupError("找不到這次互動紀錄")
        if session["status"] != "active":
            raise LookupError("這次互動已經結束")

        prior_events = self.repository.list_events(session_id)
        analysis = self.rule_engine.analyze(
            speaker=speaker,
            text=text,
            pause_before=pause_before,
            gaze_on_target=gaze_on_target,
            prior_events=prior_events,
            gaze_available=gaze_available,
        )
        if metadata is not None:
            analysis["transcription"] = metadata

        asd_observation = None
        if self.asd_analysis_service:
            asd_observation = self.asd_analysis_service.get_latest(session_id)
            if asd_observation:
                analysis["asd_v4_observation"] = asd_observation

        coach_pending = False
        if self.context_builder and self.coach_provider:
            current_event = {
                "speaker": speaker,
                "text": text,
                "pause_before": pause_before,
                "gaze_available": gaze_available,
                "gaze_on_target": gaze_on_target,
            }
            context = self.context_builder.build(
                session=session,
                events=prior_events,
                current_event=current_event,
                rule_analysis=analysis,
                asd_observation=asd_observation,
            )
            if defer_coach and callable(
                getattr(self.coach_provider, "fallback", None)
            ):
                analysis["suggestion"] = self.coach_provider.fallback(
                    context=context,
                    fallback=analysis["suggestion"],
                )
                coach_pending = bool(
                    getattr(self.coach_provider, "enabled", False)
                    and analysis["suggestion"].get("response_mode")
                    != "safety_check"
                )
            else:
                analysis["suggestion"] = self.coach_provider.generate(
                    context=context,
                    fallback=analysis["suggestion"],
                )

        event = self.repository.add_event(
            session_id=session_id,
            speaker=speaker,
            text=text,
            pause_before=pause_before,
            gaze_on_target=gaze_on_target,
            analysis=analysis,
        )

        events = [*prior_events, event]
        metrics = self.rule_engine.summarize(events)
        self.repository.update_metrics(session_id, metrics)

        return {
            "event": event,
            "metrics": metrics,
            "suggestion": analysis["suggestion"],
            "coach_source": analysis["suggestion"].get(
                "source",
                "rule_engine",
            ),
            "coach_pending": coach_pending,
        }

    def refine_event_coach(self, session_id, event_id):
        """Refine one already-saved event without blocking later events."""

        if not self.context_builder or not self.coach_provider:
            raise LookupError("教練模型尚未設定")

        with self._session_lock(session_id):
            session = self.repository.get_session(session_id)
            if not session:
                raise LookupError("找不到這次互動紀錄")
            events = self.repository.list_events(session_id)
            event_index = next(
                (
                    index
                    for index, event in enumerate(events)
                    if int(event["id"]) == int(event_id)
                ),
                None,
            )
            if event_index is None:
                raise LookupError("找不到指定的對話事件")
            event = events[event_index]
            analysis = dict(event["analysis"])
            current_event = {
                "speaker": event["speaker"],
                "text": event["text"],
                "pause_before": event["pause_before"],
                "gaze_available": analysis.get("gaze_available", False),
                "gaze_on_target": event["gaze_on_target"],
            }
            context = self.context_builder.build(
                session=session,
                events=events[:event_index],
                current_event=current_event,
                rule_analysis=analysis,
                asd_observation=analysis.get("asd_v4_observation"),
            )
            fallback = dict(analysis["suggestion"])

        # Ollama may take many seconds. Do not hold the session lock here, so
        # the next utterance can be transcribed and stored immediately.
        suggestion = self.coach_provider.generate(
            context=context,
            fallback=fallback,
        )

        with self._session_lock(session_id):
            latest_event = self.repository.get_event(event_id)
            if not latest_event or int(latest_event["session_id"]) != int(session_id):
                raise LookupError("找不到指定的對話事件")
            latest_analysis = dict(latest_event["analysis"])
            latest_analysis["suggestion"] = suggestion
            updated_event = self.repository.update_event_analysis(
                event_id,
                latest_analysis,
            )

        return {
            "event": updated_event,
            "suggestion": suggestion,
            "coach_source": suggestion.get("source", "rule_engine"),
            "coach_pending": False,
        }

    def finish_session(self, session_id):
        with self._session_lock(session_id):
            session = self.repository.finish_session(session_id)
            if not session:
                raise LookupError("找不到這次互動紀錄")
            if self.asd_analysis_service:
                self.asd_analysis_service.finish_session(session_id)
            return session
