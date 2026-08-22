import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "bernini" / "run_v2v.sh"


def find_bash():
    if os.name == "nt":
        git = shutil.which("git")
        if git is not None:
            git_bash = Path(git).resolve().parent.parent / "bin" / "bash.exe"
            if git_bash.is_file():
                return str(git_bash)
    return shutil.which("bash")


class RunV2VScriptTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bash = find_bash()
        if cls.bash is None:
            raise unittest.SkipTest("bash is required to test run_v2v.sh")

    def run_script(self, case_path=None):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            log_path = temp_path / "torchrun.log"
            stub_path = temp_path / "torchrun"
            stub_path.write_text(
                """#!/usr/bin/env bash
while (( $# )); do
    if [[ "$1" == "--case" ]]; then
        printf '%s\\n' "$2" >> "$TORCHRUN_LOG"
        exit 0
    fi
    shift
done
exit 1
""",
                encoding="utf-8",
            )
            stub_path.chmod(stub_path.stat().st_mode | stat.S_IXUSR)

            env = os.environ.copy()
            env["PATH"] = os.pathsep.join((str(temp_path), env["PATH"]))
            env["TORCHRUN_LOG"] = str(log_path)
            if case_path is None:
                env.pop("CASE_PATH", None)
            else:
                env["CASE_PATH"] = case_path

            subprocess.run(
                [self.bash, str(SCRIPT)],
                cwd=REPO_ROOT,
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )
            return log_path.read_text(encoding="utf-8").splitlines()

    def test_case_path_override_runs_only_selected_case(self):
        self.assertEqual(
            self.run_script("custom/case.json"),
            ["custom/case.json"],
        )

    def test_unset_case_path_runs_all_bundled_cases(self):
        self.assertEqual(
            self.run_script(),
            [
                "assets/testcases/v2v/v2v_case1.json",
                "assets/testcases/v2v/v2v_case2.json",
                "assets/testcases/v2v/v2v_case3.json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
