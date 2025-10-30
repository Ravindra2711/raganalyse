import ast
from typing import Tuple

FORBIDDEN_MODULES = {"os", "subprocess", "socket", "sys", "fcntl", "pty", "resource", "signal"}


def is_safe_python(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    return False, f"forbidden_import: {alias.name}"
        if isinstance(node, ast.ImportFrom):
            if node.module and node.module.split(".")[0] in FORBIDDEN_MODULES:
                return False, f"forbidden_import_from: {node.module}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if getattr(node.func, "attr", "") in {"system", "popen", "fork", "execve"}:
                return False, f"forbidden_call: {node.func.attr}"

    return True, "ok"




