"""Check that the package-owned live runtime imports without developer scripts."""

import os
import subprocess
import sys
from pathlib import Path


def test_live_runtime_imports_with_developer_modules_blocked(tmp_path: Path) -> None:
    source = """
import importlib
import importlib.abc
import pkgutil
import sys

class RejectDeveloperImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "scripts" or fullname.startswith("scripts."):
            raise AssertionError(f"Private developer import: {fullname}")
        return None

sys.meta_path.insert(0, RejectDeveloperImports())
import marl_battlegrounds._battle_client as runtime
for module in pkgutil.iter_modules(runtime.__path__, runtime.__name__ + "."):
    importlib.import_module(module.name)
assert not any(name == "scripts" or name.startswith("scripts.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
