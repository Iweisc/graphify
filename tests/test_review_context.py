from __future__ import annotations

import json

import networkx as nx
from networkx.readwrite import json_graph

import graphify.__main__ as mainmod
from graphify.review_context import build_review_context


def _build_graph():
    G = nx.DiGraph()
    G.add_node("api_file", label="api.py", source_file="src/api.py", community=0)
    G.add_node("api_handler", label="handle_pr", source_file="src/api.py", community=0)
    G.add_node("service_file", label="service.py", source_file="src/service.py", community=0)
    G.add_node("service_fn", label="plan_context", source_file="src/service.py", community=0)
    G.add_node("db_file", label="db.py", source_file="src/db.py", community=1)
    G.add_node("db_fn", label="load_graph", source_file="src/db.py", community=1)
    G.add_node("docs_file", label="design.md", source_file="docs/design.md", community=2)

    G.add_edge("api_handler", "service_fn", relation="calls", confidence="EXTRACTED")
    G.add_edge("service_fn", "db_fn", relation="imports", confidence="EXTRACTED")
    G.add_edge("service_fn", "docs_file", relation="documents", confidence="EXTRACTED")
    G.add_edge("api_file", "service_file", relation="imports", confidence="EXTRACTED")
    G.add_edge("service_file", "db_file", relation="imports", confidence="EXTRACTED")
    return G


def test_build_review_context_ranks_code_neighbors_before_docs():
    payload = build_review_context(_build_graph(), ["src/api.py"], max_related_files=3)

    related_paths = [item["path"] for item in payload["related_files"]]
    assert related_paths[:2] == ["src/service.py", "src/db.py"]
    assert "docs/design.md" in related_paths
    assert payload["summary"]["seed_nodes"] >= 1
    assert any(link["from"] == "src/api.py" and link["to"] == "src/service.py" for link in payload["graph_links"])


def test_review_context_cli_reads_github_style_changed_files_json(monkeypatch, tmp_path, capsys):
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps(json_graph.node_link_data(_build_graph(), edges="links")), encoding="utf-8")
    changed_files_path = tmp_path / "changed_files.json"
    changed_files_path.write_text(
        json.dumps([{"filename": "src/api.py"}], indent=2), encoding="utf-8"
    )

    monkeypatch.setattr(mainmod, "_check_skill_version", lambda _: None)
    monkeypatch.setattr(
        mainmod.sys,
        "argv",
        [
            "graphify",
            "review-context",
            "--graph",
            str(graph_path),
            "--changed-files-json",
            str(changed_files_path),
        ],
    )

    try:
        mainmod.main()
    except SystemExit as exc:
        assert exc.code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["changed_files"] == ["src/api.py"]
    assert payload["related_files"][0]["path"] == "src/service.py"


def test_review_context_normalizes_absolute_graph_source_paths(tmp_path):
    repo_root = tmp_path / "repo"
    out_dir = repo_root / "graphify-out"
    out_dir.mkdir(parents=True)

    G = nx.DiGraph()
    G.add_node("api_file", label="api.py", source_file=str(repo_root / "src/api.py"), community=0)
    G.add_node("service_file", label="service.py", source_file=str(repo_root / "src/service.py"), community=0)
    G.add_edge("api_file", "service_file", relation="imports", confidence="EXTRACTED")

    graph_path = out_dir / "graph.json"
    graph_path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")

    from graphify.review_context import load_graph

    payload = build_review_context(load_graph(graph_path), ["src/api.py"], max_related_files=3)
    assert payload["related_files"][0]["path"] == "src/service.py"


def test_review_context_normalizes_absolute_graph_source_paths(tmp_path):
    repo_root = tmp_path / "repo"
    out_dir = repo_root / "graphify-out"
    out_dir.mkdir(parents=True)

    G = nx.DiGraph()
    G.add_node("api_file", label="api.py", source_file=str(repo_root / "src/api.py"), community=0)
    G.add_node("service_file", label="service.py", source_file=str(repo_root / "src/service.py"), community=0)
    G.add_edge("api_file", "service_file", relation="imports", confidence="EXTRACTED")

    graph_path = out_dir / "graph.json"
    graph_path.write_text(json.dumps(json_graph.node_link_data(G, edges="links")), encoding="utf-8")

    from graphify.review_context import load_graph

    payload = build_review_context(load_graph(graph_path), ["src/api.py"], max_related_files=3)
    assert payload["related_files"][0]["path"] == "src/service.py"
