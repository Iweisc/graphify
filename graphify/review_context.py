"""PR-review-oriented context selection from graphify graph.json.

This module is intentionally narrower than graphify's repo-wide reporting flow.
It takes a set of changed files from a pull request and returns a compact,
machine-readable context pack optimized for downstream review agents.
"""
from __future__ import annotations

import argparse
import heapq
import json
from collections import Counter
from pathlib import Path
from typing import Any

from networkx.readwrite import json_graph


DEFAULT_MAX_DEPTH = 4
DEFAULT_MAX_RELATED_FILES = 16
DEFAULT_MAX_GRAPH_LINKS = 32
DEFAULT_MAX_VISITED_NODES = 1200


def normalize_path(value: str) -> str:
    return value.replace("\\", "/").replace("./", "").strip().strip("/")


def relation_weight(relation: str, direction: str) -> float:
    relation = (relation or "").lower()
    if relation in {"imports", "imports_from", "calls", "uses", "references", "depends_on"}:
        return 0.9 if direction == "out" else 0.78
    if relation in {"contains", "declares", "defines", "exports", "implements"}:
        return 0.62 if direction == "out" else 0.48
    if relation in {"rationale_for", "documents", "describes", "explains"}:
        return 0.46 if direction == "out" else 0.38
    if relation == "semantically_similar_to":
        return 0.34
    return 0.52 if direction == "out" else 0.44


def _normalize_source_path(value: str, repo_root: Path) -> str:
    path_obj = Path(str(value or ""))
    if path_obj.is_absolute():
        try:
            return normalize_path(str(path_obj.resolve().relative_to(repo_root.resolve())))
        except Exception:
            return normalize_path(path_obj.as_posix())
    return normalize_path(str(value or ""))


def load_graph(graph_path: Path):
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    if "links" not in raw and "edges" in raw:
        raw = dict(raw, links=raw["edges"])
    raw = {**raw, "directed": True}
    try:
        G = json_graph.node_link_graph(raw, edges="links")
    except TypeError:
        G = json_graph.node_link_graph(raw)
    repo_root = graph_path.resolve().parent.parent
    for _, data in G.nodes(data=True):
        if data.get("source_file"):
            data["source_file"] = _normalize_source_path(str(data.get("source_file")), repo_root)
    return G


def _node_path(data: dict[str, Any]) -> str:
    return normalize_path(str(data.get("source_file") or ""))


def _node_label(node_id: str, data: dict[str, Any]) -> str:
    return str(data.get("label") or node_id)


def _node_community(data: dict[str, Any]) -> int | None:
    value = data.get("community")
    return value if isinstance(value, int) else None


def _iter_neighbors(G, node_id: str):
    if G.is_directed():
        for _, target, edge in G.out_edges(node_id, data=True):
            yield target, edge, "out"
        for source, _, edge in G.in_edges(node_id, data=True):
            yield source, edge, "in"
    else:
        for source, target, edge in G.edges(node_id, data=True):
            neighbor = target if source == node_id else source
            yield neighbor, edge, "out"


def _extract_changed_files_from_json(json_path: Path) -> list[str]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("changed-files-json must contain a JSON list")
    paths: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("filename") or item.get("path") or item.get("file")
        if isinstance(raw_path, str) and raw_path.strip():
            paths.append(normalize_path(raw_path))
    return paths


def build_review_context(
    G,
    changed_files: list[str],
    *,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_related_files: int = DEFAULT_MAX_RELATED_FILES,
    max_graph_links: int = DEFAULT_MAX_GRAPH_LINKS,
    max_visited_nodes: int = DEFAULT_MAX_VISITED_NODES,
) -> dict[str, Any]:
    changed_files = [normalize_path(path) for path in changed_files if normalize_path(path)]
    changed_set = set(changed_files)
    node_data = dict(G.nodes(data=True))

    seed_nodes = [node_id for node_id, data in node_data.items() if _node_path(data) in changed_set]
    seed_communities = {
        community
        for node_id in seed_nodes
        for community in [_node_community(node_data[node_id])]
        if community is not None
    }

    if not seed_nodes:
        return {
            "changed_files": changed_files,
            "summary": {
                "seed_nodes": 0,
                "visited_nodes": 0,
                "max_depth": max_depth,
                "max_related_files": max_related_files,
            },
            "related_files": [],
            "graph_links": [],
            "community_hints": [],
        }

    best_score: dict[str, float] = {node_id: 1.0 for node_id in seed_nodes}
    best_depth: dict[str, int] = {node_id: 0 for node_id in seed_nodes}
    parent: dict[str, tuple[str, str]] = {}
    frontier: list[tuple[float, int, str]] = [(-1.0, 0, node_id) for node_id in seed_nodes]
    heapq.heapify(frontier)

    visited_nodes = 0
    while frontier and visited_nodes < max_visited_nodes:
        neg_score, depth, node_id = heapq.heappop(frontier)
        score = -neg_score
        if score + 1e-9 < best_score.get(node_id, 0):
            continue
        if depth > best_depth.get(node_id, depth):
            continue
        visited_nodes += 1
        if depth >= max_depth:
            continue

        for neighbor, edge, direction in _iter_neighbors(G, node_id):
            weight = relation_weight(str(edge.get("relation") or ""), direction)
            next_score = score * weight
            if next_score < 0.08:
                continue
            next_depth = depth + 1
            prev_score = best_score.get(neighbor, 0.0)
            prev_depth = best_depth.get(neighbor, 10**9)
            if next_score > prev_score * 1.03 or next_depth < prev_depth:
                best_score[neighbor] = max(prev_score, next_score)
                best_depth[neighbor] = min(prev_depth, next_depth)
                parent[neighbor] = (node_id, str(edge.get("relation") or "related_to"))
                heapq.heappush(frontier, (-next_score, next_depth, neighbor))

    file_scores: dict[str, float] = {}
    file_depths: dict[str, int] = {}
    file_via: dict[str, list[str]] = {}
    file_communities: dict[str, int | None] = {}
    file_relations: dict[str, Counter[str]] = {}

    for node_id, score in best_score.items():
        data = node_data.get(node_id, {})
        path = _node_path(data)
        if not path or path in changed_set:
            continue
        community = _node_community(data)
        adjusted = score
        if community is not None and community in seed_communities:
            adjusted += 0.08
        if adjusted > file_scores.get(path, 0.0):
            file_scores[path] = adjusted
            file_depths[path] = best_depth.get(node_id, max_depth)
            file_communities[path] = community
            # Build provenance chain
            trace: list[str] = []
            cursor = node_id
            seen: set[str] = set()
            while cursor in parent and cursor not in seen and len(trace) < 6:
                seen.add(cursor)
                prev, relation = parent[cursor]
                prev_data = node_data.get(prev, {})
                trace.append(f"{_node_label(prev, prev_data)} --{relation}--> {_node_label(cursor, data if cursor == node_id else node_data.get(cursor, {}))}")
                cursor = prev
            file_via[path] = list(reversed(trace))
        rel_counter = file_relations.setdefault(path, Counter())
        if node_id in parent:
            rel_counter[parent[node_id][1]] += 1

    related_files = [
        {
            "path": path,
            "score": round(score, 4),
            "depth": file_depths.get(path, max_depth),
            "community": file_communities.get(path),
            "top_relations": [name for name, _ in file_relations.get(path, Counter()).most_common(3)],
            "via": file_via.get(path, []),
        }
        for path, score in sorted(file_scores.items(), key=lambda item: item[1], reverse=True)[:max_related_files]
    ]

    included_files = changed_set | {item["path"] for item in related_files}
    graph_link_map: dict[tuple[str, str, str], float] = {}
    for source, target, edge in G.edges(data=True):
        source_data = node_data.get(source, {})
        target_data = node_data.get(target, {})
        source_path = _node_path(source_data)
        target_path = _node_path(target_data)
        if not source_path or not target_path or source_path == target_path:
            continue
        if source_path not in included_files or target_path not in included_files:
            continue
        relation = str(edge.get("relation") or "related_to")
        key = (source_path, target_path, relation)
        score = best_score.get(source, 0.0) + best_score.get(target, 0.0)
        if score > graph_link_map.get(key, 0.0):
            graph_link_map[key] = score

    graph_links = [
        {
            "from": source_path,
            "to": target_path,
            "relation": relation,
            "score": round(score, 4),
        }
        for (source_path, target_path, relation), score in sorted(
            graph_link_map.items(), key=lambda item: item[1], reverse=True
        )[:max_graph_links]
    ]

    community_counts = Counter(
        community
        for path, community in file_communities.items()
        if path in {item["path"] for item in related_files} and community is not None
    )
    community_hints = [
        {"community": community, "related_file_count": count}
        for community, count in community_counts.most_common(6)
    ]

    return {
        "changed_files": changed_files,
        "summary": {
            "seed_nodes": len(seed_nodes),
            "visited_nodes": visited_nodes,
            "max_depth": max_depth,
            "max_related_files": max_related_files,
        },
        "related_files": related_files,
        "graph_links": graph_links,
        "community_hints": community_hints,
    }


def cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="graphify review-context")
    parser.add_argument("--graph", default="graphify-out/graph.json")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--changed-files-json")
    parser.add_argument("--max-depth", type=int, default=DEFAULT_MAX_DEPTH)
    parser.add_argument("--max-related-files", type=int, default=DEFAULT_MAX_RELATED_FILES)
    parser.add_argument("--max-graph-links", type=int, default=DEFAULT_MAX_GRAPH_LINKS)
    parser.add_argument("--max-visited-nodes", type=int, default=DEFAULT_MAX_VISITED_NODES)
    args = parser.parse_args(argv)

    changed_files = [normalize_path(item) for item in args.changed_file if normalize_path(item)]
    if args.changed_files_json:
        changed_files.extend(_extract_changed_files_from_json(Path(args.changed_files_json)))
    changed_files = list(dict.fromkeys(changed_files))
    if not changed_files:
        parser.error("provide at least one --changed-file or --changed-files-json")

    graph_path = Path(args.graph).resolve()
    if not graph_path.exists():
        parser.error(f"graph file not found: {graph_path}")

    G = load_graph(graph_path)
    payload = build_review_context(
        G,
        changed_files,
        max_depth=max(1, args.max_depth),
        max_related_files=max(1, args.max_related_files),
        max_graph_links=max(1, args.max_graph_links),
        max_visited_nodes=max(50, args.max_visited_nodes),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0
