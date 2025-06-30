import ast
import json
from pathlib import Path
import networkx as nx


def parse_module(path: Path):
    module_name = path.with_suffix("").as_posix().replace("/", ".")
    with open(path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(path))

    classes = []
    functions = []
    imports = []
    calls = {}
    class_bases = {}
    docstrings = {"module": ast.get_docstring(tree)}
    signatures = {}

    for node in tree.body:
        if isinstance(node, ast.Import):
            for n in node.names:
                imports.append(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)
        elif isinstance(node, ast.ClassDef):
            classes.append(node.name)
            docstrings[node.name] = ast.get_docstring(node)
            bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
            class_bases[node.name] = bases
            calls[node.name] = []
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        calls[node.name].append(n.func.id)
                    elif isinstance(n.func, ast.Attribute):
                        calls[node.name].append(n.func.attr)
        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            docstrings[node.name] = ast.get_docstring(node)
            
            # Extract signature
            args = []
            for arg in node.args.args:
                arg_info = arg.arg
                if arg.annotation:
                    # Attempt to reconstruct the type annotation string
                    try:
                        arg_info += f": {ast.unparse(arg.annotation)}"
                    except:
                        arg_info += ": complex_type"
                args.append(arg_info)
            signatures[node.name] = f"({', '.join(args)})"

            calls[node.name] = []
            for n in ast.walk(node):
                if isinstance(n, ast.Call):
                    if isinstance(n.func, ast.Name):
                        calls[node.name].append(n.func.id)
                    elif isinstance(n.func, ast.Attribute):
                        calls[node.name].append(n.func.attr)

    return {
        "module": module_name,
        "classes": classes,
        "functions": functions,
        "imports": imports,
        "calls": calls,
        "bases": class_bases,
        "docstrings": docstrings,
        "signatures": signatures,
    }


def build_repo_graph(root_dirs, graphml_path="overview/knowledge_graph.graphml", json_path="overview/knowledge_graph.json"):
    modules = {}
    for d in root_dirs:
        for path in Path(d).rglob("*.py"):
            modules[path] = parse_module(path)

    G = nx.DiGraph()

    # add nodes
    for info in modules.values():
        doc = info["docstrings"].get("module")
        G.add_node(info["module"], type="module", doc=doc or "")
        for cls in info["classes"]:
            doc = info["docstrings"].get(cls)
            G.add_node(f"{info['module']}.{cls}", type="class", doc=doc or "")
        for fn in info["functions"]:
            doc = info["docstrings"].get(fn)
            sig = info["signatures"].get(fn)
            G.add_node(f"{info['module']}.{fn}", type="function", doc=doc or "", signature=sig)

    # edges for imports
    for info in modules.values():
        for imp in info["imports"]:
            if imp in [m["module"] for m in modules.values()]:
                G.add_edge(info["module"], imp, type="import")

    # inheritance edges
    for info in modules.values():
        for cls, bases in info["bases"].items():
            for b in bases:
                for mod_info in modules.values():
                    if b in mod_info["classes"]:
                        G.add_edge(f"{mod_info['module']}.{b}", f"{info['module']}.{cls}", type="inherit")

    # function call edges
    name_to_module = {}
    for info in modules.values():
        for fn in info["functions"]:
            name_to_module[fn] = info["module"]
        for cls in info["classes"]:
            name_to_module[cls] = info["module"]

    for info in modules.values():
        for func, calls in info["calls"].items():
            src = f"{info['module']}.{func}"
            for c in calls:
                if c in name_to_module:
                    dst = f"{name_to_module[c]}.{c}"
                    G.add_edge(src, dst, type="call")

    nx.write_graphml(G, graphml_path)
    with open(json_path, "w", encoding="utf-8") as f:
        data = {
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
            "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
        }
        json.dump(data, f, indent=2)

    return graphml_path, json_path


if __name__ == "__main__":
    build_repo_graph(["alignment", "htr_base"])
