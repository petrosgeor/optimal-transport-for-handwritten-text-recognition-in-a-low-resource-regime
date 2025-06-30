# Repository Knowledge Graph

This repository includes a script that analyses the Python source under
`alignment/` and `htr_base/` to build a graph describing relationships between
modules, classes and functions.

## How it works

1. Each `.py` file is parsed using `ast` to extract:
   - module name
   - classes and their bases
   - function definitions
   - imported modules
   - function calls within the module
2. A directed graph is created using `networkx` where modules, classes and
   functions are nodes. Edges represent imports, inheritance and direct
   function calls.
3. The graph is written to two files:
   - `knowledge_graph.graphml` – suitable for graph viewers
   - `knowledge_graph.json` – machine readable node and edge data

Run the script with:

```bash
python scripts/generate_knowledge_graph.py
```

## Key nodes

`HTRNet` and `Projector` from `htr_base.models` appear as central classes.
Functions such as `refine_visual_backbone` and `train_projector` in
`alignment.alignment_trainer` have many outgoing calls, linking training
utilities with dataset helpers from `htr_base.utils`.

## Notable connections

- `alignment.alignment_trainer` imports and calls functions from
  `alignment.alignment_utilities` and `alignment.losses` to orchestrate the
  training pipeline.
- Modules under `alignment` rely heavily on components defined in `htr_base`,
  particularly `HTRNet` and dataset utilities.

The resulting graph highlights these interactions and can be visualised using
Graphviz or any tool that supports GraphML.
