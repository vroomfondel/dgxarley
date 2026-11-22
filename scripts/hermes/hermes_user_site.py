import os
import sys

path = os.path.join(
    os.environ.get("HERMES_PIP_TARGET_BASE", "/opt/data/.local/pip"), "python%d.%d" % sys.version_info[:2]
)

# Root must not create it: a root-owned dir on /opt/data would lock the hermes user out.
if os.geteuid() != 0 and os.path.isdir("/opt/data"):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass

if os.path.isdir(path) and path not in sys.path:
    sys.path.append(path)
