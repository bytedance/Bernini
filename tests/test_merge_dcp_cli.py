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

import ast
from pathlib import Path
import unittest


class MergeDcpCliTest(unittest.TestCase):
    def test_shard_size_help_matches_default(self):
        source_path = Path(__file__).parents[1] / "tools" / "merge_dcp_to_hf_pt.py"
        tree = ast.parse(source_path.read_text(encoding="utf-8"))

        shard_argument = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--shard-size"
        )
        keywords = {keyword.arg: keyword.value for keyword in shard_argument.keywords}
        default = ast.literal_eval(keywords["default"])
        help_text = ast.literal_eval(keywords["help"])

        self.assertEqual(default, 5_000_000_000)
        self.assertIn("default: 5GB", help_text)


if __name__ == "__main__":
    unittest.main()
