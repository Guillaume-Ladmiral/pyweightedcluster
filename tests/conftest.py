import importlib.util
import sys
from pathlib import Path

pytest_plugins = ["tests.r_parity_utils", "tests.traminex_parity_utils"]

# Ensure src/ is importable when running tests from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure we import the local package implementation, not any similarly-named project elsewhere
# We explicitly load the package from src/weightedcluster/__init__.py
local_pkg_dir = SRC_DIR / "weightedcluster"
local_init = local_pkg_dir / "__init__.py"
for module_name in tuple(sys.modules):
    if module_name == "weightedcluster" or module_name.startswith("weightedcluster."):
        del sys.modules[module_name]

spec = importlib.util.spec_from_file_location(
    "weightedcluster", local_init, submodule_search_locations=[str(local_pkg_dir)]
)
if spec is None or spec.loader is None:
    raise ImportError("could not create spec for local weightedcluster package")
module = importlib.util.module_from_spec(spec)
sys.modules["weightedcluster"] = module
spec.loader.exec_module(module)
