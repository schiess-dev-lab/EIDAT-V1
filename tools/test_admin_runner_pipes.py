import os
import sys
import tempfile
import traceback
import unittest
from multiprocessing import get_context
from pathlib import Path
from unittest import mock


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _worker_run_subprocess_json(queue) -> None:  # pragma: no cover (runs in child process)
    try:
        root = _repo_root()
        sys.path.insert(0, str(root / "EIDAT_App_Files"))
        from Production import admin_runner  # type: ignore

        code = (
            "import json,sys\n"
            "sys.stderr.write('hello stderr\\n'); sys.stderr.flush()\n"
            "sys.stdout.write(json.dumps({'data':'a'*70000}, separators=(',',':')))\n"
            "sys.stdout.write('\\n'); sys.stdout.flush()\n"
        )
        argv = [sys.executable, "-c", code]

        logged: list[str] = []

        def _on_log(line: str) -> None:
            logged.append(str(line))

        payload = admin_runner._run_subprocess_json(  # type: ignore[attr-defined]
            argv,
            env=os.environ.copy(),
            cmd="test",
            on_log=_on_log,
        )
        queue.put(
            {
                "ok": True,
                "logged": len(logged),
                "data_len": len(str(payload.get("data") or "")),
            }
        )
    except Exception:
        queue.put({"ok": False, "error": traceback.format_exc()})


class TestAdminRunnerPipes(unittest.TestCase):
    def test_run_subprocess_json_does_not_deadlock(self) -> None:
        ctx = get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_worker_run_subprocess_json, args=(q,))
        p.start()
        p.join(15)
        if p.is_alive():
            p.terminate()
            p.join(5)
            self.fail("admin_runner subprocess capture appears to have deadlocked (timeout).")

        self.assertEqual(p.exitcode, 0)
        res = q.get(timeout=2)
        if not res.get("ok"):
            self.fail(str(res.get("error") or "unknown failure"))

        self.assertGreaterEqual(int(res.get("logged") or 0), 1)
        self.assertGreaterEqual(int(res.get("data_len") or 0), 70000)


class TestScanIgnoresHeavyDirs(unittest.TestCase):
    def test_scan_prunes_dot_git_and_node_modules(self) -> None:
        import tempfile

        root = _repo_root()
        sys.path.insert(0, str(root / "EIDAT_App_Files" / "Application"))
        from eidat_manager_db import support_paths  # type: ignore
        from eidat_manager_scan import scan_global_repo  # type: ignore

        # On Windows + SQLite(WAL), file locks can briefly persist after closing; don't fail test cleanup.
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            repo = Path(td)
            (repo / "normal").mkdir(parents=True, exist_ok=True)
            (repo / ".git").mkdir(parents=True, exist_ok=True)
            (repo / "node_modules").mkdir(parents=True, exist_ok=True)
            (repo / "normal" / "a.pdf").write_text("x", encoding="utf-8")
            (repo / ".git" / "b.pdf").write_text("x", encoding="utf-8")
            (repo / "node_modules" / "c.pdf").write_text("x", encoding="utf-8")

            paths = support_paths(repo)
            summary = scan_global_repo(paths)

            self.assertEqual(int(summary.pdf_count or 0), 1)
            self.assertEqual(len(summary.candidates), 1)
            self.assertEqual(summary.candidates[0].rel_path.replace("\\", "/"), "normal/a.pdf")


class TestSingleFileForceUpdate(unittest.TestCase):
    def test_backend_process_passes_file_filters_to_manager(self) -> None:
        root = _repo_root()
        sys.path.insert(0, str(root / "EIDAT_App_Files"))
        from ui_next import backend as be  # type: ignore

        with mock.patch.object(
            be,
            "_run_eidat_manager",
            return_value={"processed_ok": 0, "processed_failed": 0, "results": []},
        ) as run_mock:
            be.eidat_manager_process(
                Path("repo"),
                force=True,
                file_paths=[Path("data") / "a.pdf", "nested/b.xlsx"],
            )

        run_mock.assert_called_once()
        _, cmd, args = run_mock.call_args.args
        self.assertEqual(cmd, "process")
        self.assertIn("--force", args)
        self.assertEqual(args.count("--file"), 2)
        self.assertIn(str(Path("data") / "a.pdf"), args)
        self.assertIn("nested/b.xlsx", args)

    def test_admin_runner_force_single_file_processes_only_selected_rel_path(self) -> None:
        root = _repo_root()
        sys.path.insert(0, str(root / "EIDAT_App_Files"))
        from Production import admin_runner  # type: ignore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            base = Path(td)
            node = base / "node"
            runtime = base / "runtime"
            source = node / "data" / "a.pdf"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(b"%PDF-1.4\n")
            (runtime / "EIDAT_App_Files").mkdir(parents=True, exist_ok=True)

            calls: list[tuple[str, list[str]]] = []

            def fake_run_manager(_py, _runtime, _node, cmd, extra=None, **_kwargs):
                calls.append((str(cmd), list(extra or [])))
                if cmd == "scan":
                    return {"candidates_count": 3}
                if cmd == "process":
                    return {
                        "processed_ok": 1,
                        "processed_failed": 0,
                        "results": [{"rel_path": "data/a.pdf", "ok": True}],
                    }
                if cmd == "index":
                    return {"indexed_count": 1}
                return {}

            with (
                mock.patch.object(admin_runner, "_run_manager", side_effect=fake_run_manager),
                mock.patch.object(admin_runner, "_load_node_config", return_value={}),
            ):
                result = admin_runner.run_force_single_file(
                    node_root=node,
                    runtime_root=runtime,
                    file_path=source,
                )

            self.assertTrue(result.ok, result.error)
            process_calls = [extra for cmd, extra in calls if cmd == "process"]
            self.assertEqual(process_calls, [["--force", "--file", "data/a.pdf"]])
            self.assertIn("index", [cmd for cmd, _extra in calls])


if __name__ == "__main__":
    unittest.main()
