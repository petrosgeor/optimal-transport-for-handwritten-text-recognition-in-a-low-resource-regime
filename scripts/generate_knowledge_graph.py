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
    attributes = {}
    methods = {}

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
            
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except AttributeError:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)

            class_bases[node.name] = bases
            attributes[node.name] = []
            methods[node.name] = []

            for class_node in node.body:
                if isinstance(class_node, ast.FunctionDef):  # Method
                    method_name = class_node.name
                    methods[node.name].append(method_name)
                    
                    full_method_name = f"{node.name}.{method_name}"
                    docstrings[full_method_name] = ast.get_docstring(class_node)

                    # Signature
                    args = []
                    for arg in class_node.args.args:
                        arg_info = arg.arg
                        if arg.annotation:
                            try:
                                arg_info += f": {ast.unparse(arg.annotation)}"
                            except:
                                arg_info += ": complex_type"
                        args.append(arg_info)
                    signatures[full_method_name] = f"({', '.join(args)})"

                    # Calls from method
                    method_calls = []
                    for n in ast.walk(class_node):
                        if isinstance(n, ast.Call):
                            if isinstance(n.func, ast.Name):
                                method_calls.append(n.func.id)
                            elif isinstance(n.func, ast.Attribute):
                                method_calls.append(n.func.attr)
                    calls[full_method_name] = method_calls

                    # Attributes defined in method
                    for n in ast.walk(class_node):
                        if isinstance(n, ast.Assign):
                            for target in n.targets:
                                if (isinstance(target, ast.Attribute) and
                                        isinstance(target.value, ast.Name) and
                                        target.value.id == 'self'):
                                    attr_name = target.attr
                                    instance_of = None
                                    if isinstance(n.value, ast.Call):
                                        if isinstance(n.value.func, ast.Name):
                                            instance_of = n.value.func.id
                                        elif isinstance(n.value.func, ast.Attribute):
                                            instance_of = n.value.func.attr
                                    attr_info = {"name": attr_name, "instance_of": instance_of}
                                    if attr_info not in attributes[node.name]:
                                        attributes[node.name].append(attr_info)

        elif isinstance(node, ast.FunctionDef):
            functions.append(node.name)
            docstrings[node.name] = ast.get_docstring(node)
            
            # Extract signature
            args = []
            for arg in node.args.args:
                arg_info = arg.arg
                if arg.annotation:
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
        "attributes": attributes,
        "methods": methods,
    }


def build_repo_graph(root_dirs, graphml_path="overview/knowledge_graph.graphml", json_path="overview/knowledge_graph.json"):
    schema_version = "1.2"
    schema = {
        "node_types": {
            "module": "Represents a Python module (.py file).",
            "class": "Represents a Python class defined within the repository.",
            "external_class": "Represents a class imported from an external library.",
            "method": "Represents a method within a class.",
            "function": "Represents a standalone function in a module.",
            "attribute": "Represents a class attribute, typically defined with 'self.<attr>'."
        },
        "edge_types": {
            "import": "Source module imports target module.",
            "inherit": "Source class inherits from target class (internal or external).",
            "call": "Source function/method calls target function/method.",
            "has_method": "Source class contains target method.",
            "has_attr": "Source class has target attribute.",
            "instance_of": "Source attribute is an instance of target class."
        }
    }

    modules = {}
    for d in root_dirs:
        for path in Path(d).rglob("*.py"):
            modules[path] = parse_module(path)

    G = nx.DiGraph()
    
    class_to_module = {}
    for info in modules.values():
        for cls in info["classes"]:
            class_to_module[cls] = info["module"]

    # add nodes
    for info in modules.values():
        doc = info["docstrings"].get("module")
        G.add_node(info["module"], type="module", doc=doc or "")
        for cls in info["classes"]:
            class_node_id = f"{info['module']}.{cls}"
            doc = info["docstrings"].get(cls)
            G.add_node(class_node_id, type="class", doc=doc or "")
            
            if cls in info["methods"]:
                for method in info["methods"][cls]:
                    method_node_id = f"{class_node_id}.{method}"
                    full_method_name = f"{cls}.{method}"
                    doc = info["docstrings"].get(full_method_name)
                    sig = info["signatures"].get(full_method_name)
                    G.add_node(method_node_id, type="method", doc=doc or "", signature=sig)
                    G.add_edge(class_node_id, method_node_id, type="has_method")

        for fn in info["functions"]:
            doc = info["docstrings"].get(fn)
            sig = info["signatures"].get(fn)
            G.add_node(f"{info['module']}.{fn}", type="function", doc=doc or "", signature=sig)

    # add attribute nodes and has_attr edges
    for info in modules.values():
        module_name = info["module"]
        for cls_name, attrs in info["attributes"].items():
            class_node_id = f"{module_name}.{cls_name}"
            for attr in attrs:
                attr_name = attr["name"]
                attr_node_id = f"{class_node_id}.{attr_name}"
                G.add_node(attr_node_id, type="attribute")
                G.add_edge(class_node_id, attr_node_id, type="has_attr")
                if attr['instance_of'] and attr['instance_of'] in class_to_module:
                    dst_module = class_to_module[attr['instance_of']]
                    dst_node = f"{dst_module}.{attr['instance_of']}"
                    if G.has_node(dst_node):
                        G.add_edge(attr_node_id, dst_node, type="instance_of")

    # edges for imports
    for info in modules.values():
        for imp in info["imports"]:
            if imp in [m["module"] for m in modules.values()]:
                G.add_edge(info["module"], imp, type="import")

    # inheritance edges
    for info in modules.values():
        for cls, bases in info["bases"].items():
            derived_class_node_id = f"{info['module']}.{cls}"
            for base_name in bases:
                simple_base_name = base_name.split('.')[-1]
                if simple_base_name in class_to_module:
                    base_module = class_to_module[simple_base_name]
                    base_class_node_id = f"{base_module}.{simple_base_name}"
                    if G.has_node(base_class_node_id):
                        G.add_edge(derived_class_node_id, base_class_node_id, type="inherit")
                else:
                    external_class_node_id = base_name
                    if not G.has_node(external_class_node_id):
                        G.add_node(external_class_node_id, type="external_class", doc=f"External class: {external_class_node_id}")
                    G.add_edge(derived_class_node_id, external_class_node_id, type="inherit")

    # function call edges
    name_to_module = {}
    for info in modules.values():
        for fn in info["functions"]:
            name_to_module[fn] = info["module"]
        for cls in info["classes"]:
            name_to_module[cls] = info["module"]
            
    for info in modules.values():
        for func_or_method, calls_list in info["calls"].items():
            src_node_id = None
            if "." in func_or_method: # Method
                cls_name, method_name = func_or_method.split('.', 1)
                if cls_name in class_to_module and class_to_module[cls_name] == info['module']:
                     src_node_id = f"{info['module']}.{cls_name}.{method_name}"
            else: # Function
                src_node_id = f"{info['module']}.{func_or_method}"

            if src_node_id and G.has_node(src_node_id):
                for c in calls_list:
                    if c in name_to_module:
                        dst_module = name_to_module[c]
                        dst_node_id = f"{dst_module}.{c}"
                        if G.has_node(dst_node_id):
                            G.add_edge(src_node_id, dst_node_id, type="call")

    nx.write_graphml(G, graphml_path)
    with open(json_path, "w", encoding="utf-8") as f:
        data = {
            "schema_version": schema_version,
            "schema": schema,
            "nodes": [{"id": n, **G.nodes[n]} for n in G.nodes],
            "edges": [{"source": u, "target": v, **G.edges[u, v]} for u, v in G.edges],
        }
        json.dump(data, f, indent=2)

    return graphml_path, json_path


if __name__ == "__main__":
    build_repo_graph(["alignment", "htr_base"])