import os
import sys
import traceback
import unittest
from multiprocessing import get_context
from pathlib import Path


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


if __name__ == "__main__":
    unittest.main()
