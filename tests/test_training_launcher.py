# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = Path("scripts/bernini_r_train/train_bernini_renderer.sh")
DEFAULT_CONFIG = "configs/bernini_renderer_train/train_cfg/bernini_renderer_high.yaml"
LOW_CONFIG = "configs/bernini_renderer_train/train_cfg/bernini_renderer_low.yaml"


def find_bash() -> str | None:
    candidates: list[Path] = []
    if os.name == "nt":
        git = shutil.which("git")
        if git:
            candidates.append(Path(git).resolve().parent.parent / "bin" / "bash.exe")
    if bash := shutil.which("bash"):
        candidates.append(Path(bash))

    for candidate in candidates:
        try:
            result = subprocess.run(
                [str(candidate), "--version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            continue
        if result.returncode == 0:
            return str(candidate)
    return None


class TrainingLauncherTest(unittest.TestCase):
    def run_launcher(self, *arguments: str) -> list[str]:
        bash = find_bash()
        if bash is None:
            self.skipTest("bash is required to exercise the training launcher")

        with tempfile.TemporaryDirectory() as temp_dir:
            bin_dir = Path(temp_dir, "bin")
            bin_dir.mkdir()
            trace_file = Path(temp_dir, "torchrun-args.txt")

            self.write_stub(
                bin_dir / "python",
                "#!/usr/bin/env bash\ncat >/dev/null\n",
            )
            self.write_stub(
                bin_dir / "torchrun",
                '#!/usr/bin/env bash\nprintf \'%s\\n\' "$@" > "$TRACE_FILE"\n',
            )

            environment = os.environ.copy()
            environment["PATH"] = os.pathsep.join(
                [str(bin_dir), environment.get("PATH", "")]
            )
            environment["TRACE_FILE"] = trace_file.as_posix()

            completed = subprocess.run(
                [bash, str(LAUNCHER), *arguments],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(0, completed.returncode, completed.stderr)
            return trace_file.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def write_stub(path: Path, contents: str) -> None:
        path.write_text(contents, encoding="utf-8", newline="\n")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)

    def test_default_config_is_followed_by_overrides(self) -> None:
        arguments = self.run_launcher("--train.max_steps", "10")

        self.assertEqual(
            [DEFAULT_CONFIG],
            [argument for argument in arguments if argument.endswith(".yaml")],
        )
        self.assertEqual(["--train.max_steps", "10"], arguments[-2:])

    def test_explicit_config_replaces_default(self) -> None:
        arguments = self.run_launcher(LOW_CONFIG, "--train.max_steps", "10")

        self.assertEqual(
            [LOW_CONFIG],
            [argument for argument in arguments if argument.endswith(".yaml")],
        )
        self.assertNotIn(DEFAULT_CONFIG, arguments)
        self.assertEqual(["--train.max_steps", "10"], arguments[-2:])


if __name__ == "__main__":
    unittest.main()
