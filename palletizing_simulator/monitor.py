class MonitorWindow:
    def __init__(self) -> None:
        import omni.ui as ui

        self.window = ui.Window("Palletizing Monitor", width=420, height=320)
        self.lines = {
            "episode": None,
            "stacking": None,
            "success": None,
            "failure": None,
            "collapse": None,
            "oob": None,
            "drop": None,
            "total": None,
            "avg_episode": None,
            "avg_stacking": None,
            "avg_success": None,
        }

        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Current Episode", height=24)
                self.lines["episode"] = ui.Label("", height=22)
                self.lines["stacking"] = ui.Label("", height=22)
                self.lines["success"] = ui.Label("", height=22)
                self.lines["failure"] = ui.Label("", height=22)
                self.lines["collapse"] = ui.Label("", height=22)
                self.lines["oob"] = ui.Label("", height=22)
                self.lines["drop"] = ui.Label("", height=22)
                self.lines["total"] = ui.Label("", height=22)

                ui.Spacer(height=8)
                ui.Label("Average Summary", height=24)
                self.lines["avg_episode"] = ui.Label("", height=22)
                self.lines["avg_stacking"] = ui.Label("", height=22)
                self.lines["avg_success"] = ui.Label("", height=22)

    def _as_float(self, value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _as_int(self, value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def update_episode(self, file_eval: dict) -> None:
        """
        지원 포맷
        1) evaluator.evaluate_file() 반환값
           - stacking_rate_pct_official
           - stacking_rate_pct_raw
        2) result.json 의 files[] 원소
           - stacking_rate_pct
           - stacking_rate_pct_raw
        """
        episode = str(file_eval.get("episode", "-"))

        official = self._as_float(
            file_eval.get("stacking_rate_pct_official", file_eval.get("stacking_rate_pct", 0.0))
        )
        raw = self._as_float(file_eval.get("stacking_rate_pct_raw", official))

        success_count = self._as_int(file_eval.get("success_count", 0))
        failure_count = self._as_int(file_eval.get("failure_count", 0))
        collapse_count = self._as_int(file_eval.get("collapse_count", 0))
        oob_count = self._as_int(file_eval.get("out_of_bounds_count", 0))
        drop_count = self._as_int(file_eval.get("drop_count", 0))
        total_boxes = self._as_int(file_eval.get("total_boxes", 0))

        source = file_eval.get("source", None)
        if source:
            self.lines["episode"].text = f'episode: "{episode}" ({source})'
        else:
            self.lines["episode"].text = f'episode: "{episode}"'

        self.lines["stacking"].text = f"stacking_rate_pct: {official:.1f} [{raw:.1f}]"
        self.lines["success"].text = f"success_count: {success_count}"
        self.lines["failure"].text = f"failure_count: {failure_count}"
        self.lines["collapse"].text = f"collapse_count: {collapse_count}"
        self.lines["oob"].text = f"out_of_bounds_count: {oob_count}"
        self.lines["drop"].text = f"drop_count: {drop_count}"
        self.lines["total"].text = f"total_boxes: {total_boxes}"

    def update_summary(self, result: dict) -> None:
        """
        evaluator.build_result() 반환 포맷 처리
        - result["summary"] 사용
        """
        sm = result.get("summary", {})

        success_episodes = self._as_int(sm.get("success_episodes", 0))
        failure_episodes = self._as_int(sm.get("failure_episodes", 0))
        avg_stacking = self._as_float(sm.get("avg_stacking_rate_pct", 0.0))
        avg_stacking_raw = self._as_float(sm.get("avg_stacking_rate_pct_raw", avg_stacking))
        avg_success = self._as_float(sm.get("avg_success_rate_pct", 0.0))

        self.lines["avg_episode"].text = (
            f"episodes: success {success_episodes} / failure {failure_episodes}"
        )
        self.lines["avg_stacking"].text = (
            f"avg_stacking_rate_pct: {avg_stacking:.1f} [{avg_stacking_raw:.1f}]"
        )
        self.lines["avg_success"].text = f"avg_success_rate_pct: {avg_success:.1f}"

    def update_from_result_file(self, result: dict, file_index: int = -1) -> None:
        """
        result.json 전체를 받아서
        - files[file_index] -> Current Episode
        - summary -> Average Summary
        둘 다 한 번에 갱신
        """
        files = result.get("files", [])
        if files:
            self.update_episode(files[file_index])
        self.update_summary(result)