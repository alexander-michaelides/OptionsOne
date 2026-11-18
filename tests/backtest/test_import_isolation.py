import ast
from importlib.util import resolve_name
from pathlib import Path


def test_backtest_import_graph_never_reaches_paper_broker() -> None:
    root = Path(__file__).resolve().parents[2]
    pending = [
        ".".join(
            path.relative_to(root).with_suffix("").parts
            if path.stem != "__init__"
            else path.relative_to(root).parent.parts
        )
        for path in (root / "optionsone" / "backtest").rglob("*.py")
    ]
    seen: set[str] = set()
    while pending:
        module = pending.pop()
        if module in seen or not module.startswith("optionsone"):
            continue
        assert module != "optionsone.execution.alpaca_paper"
        seen.add(module)
        path = root.joinpath(*module.split("."))
        package = module
        if path.is_dir():
            path /= "__init__.py"
        else:
            package = module.rpartition(".")[0]
            pending.append(package)
            path = path.with_suffix(".py")
        if not path.is_file():
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                pending.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                name = node.module or ""
                base = resolve_name("." * node.level + name, package) if node.level else name
                pending.append(base)
                pending.extend(f"{base}.{alias.name}" for alias in node.names)
