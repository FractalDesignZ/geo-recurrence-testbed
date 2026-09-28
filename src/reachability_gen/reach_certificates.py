"""Post-hoc reachability certificates under hard Â (MEASURE).

Deterministic certificates on (Â, prediction [, optional witness]).
Checker BFS is allowed ONLY for post-hoc witness / unreachable validation —
never as a training target, init, or model input feature.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

from reachability_gen.graph import adjacency_list, reachable_bfs


@dataclass(frozen=True)
class ReachCertificate:
    """Post-hoc structural certificate for one (Â, pred) pair."""

    clean: bool
    pred: int  # 0=NO / unreachable, 1=YES / reachable
    kind: str  # path_witness | unreachable_checker | dirty_yes | dirty_no | self_reach
    reason: str
    witness: Optional[tuple[int, ...]] = None  # node path s=v0..vk=t when YES clean
    reachable_by_checker: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.witness is not None:
            d["witness"] = list(self.witness)
        return d


def edge_set(edges: Sequence[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(u), int(v)) for u, v in edges}


def validate_path_witness(
    edges: Sequence[tuple[int, int]],
    path: Sequence[int],
    *,
    s: int,
    t: int,
) -> bool:
    """True iff path is a directed walk s→…→t with every consecutive edge in Â."""
    if not path:
        return False
    if int(path[0]) != int(s) or int(path[-1]) != int(t):
        return False
    if len(path) == 1:
        return int(s) == int(t)
    es = edge_set(edges)
    for i in range(len(path) - 1):
        u, v = int(path[i]), int(path[i + 1])
        if (u, v) not in es:
            return False
    return True


def find_path_witness(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
) -> Optional[tuple[int, ...]]:
    """Reconstruct a shortest directed path s→t via BFS parents, or None.

    Checker utility only — not a model feature.
    """
    s, t = int(s), int(t)
    if not (0 <= s < n and 0 <= t < n):
        raise ValueError(f"s={s} t={t} out of range for n={n}")
    if s == t:
        return (s,)
    adj = adjacency_list(n, edges)
    parent = [-1] * n
    seen = [False] * n
    q: deque[int] = deque([s])
    seen[s] = True
    found = False
    while q:
        u = q.popleft()
        for v in adj[u]:
            if not seen[v]:
                seen[v] = True
                parent[v] = u
                if v == t:
                    found = True
                    q.clear()
                    break
                q.append(v)
        if found:
            break
    if not found:
        return None
    # Reconstruct t ← … ← s
    nodes: list[int] = [t]
    cur = t
    while cur != s:
        cur = parent[cur]
        if cur < 0:
            return None
        nodes.append(cur)
    nodes.reverse()
    return tuple(nodes)


def certify_prediction(
    n: int,
    edges: Sequence[tuple[int, int]],
    s: int,
    t: int,
    pred: int,
    *,
    witness: Optional[Sequence[int]] = None,
) -> ReachCertificate:
    """Issue a post-hoc certificate for pred ∈ {0,1} under hard Â.

    YES (pred=1): clean iff a validating path-witness exists (provided or
    reconstructed by checker BFS). Else dirty_yes.
    NO (pred=0): clean iff checker confirms unreachable. Else dirty_no
    (counted dirty; caller decides policy — default fail-closed keeps pred=0).
    """
    pred = int(pred)
    s, t, n = int(s), int(t), int(n)
    if pred not in (0, 1):
        raise ValueError(f"pred must be 0 or 1, got {pred}")

    if pred == 1:
        # Prefer provided witness if it validates; else reconstruct via checker.
        if witness is not None and validate_path_witness(edges, witness, s=s, t=t):
            w = tuple(int(x) for x in witness)
            return ReachCertificate(
                clean=True,
                pred=1,
                kind="path_witness",
                reason="provided_witness_validates_on_A_hat",
                witness=w,
                reachable_by_checker=True,
            )
        reconstructed = find_path_witness(n, edges, s, t)
        if reconstructed is not None and validate_path_witness(
            edges, reconstructed, s=s, t=t
        ):
            kind = "self_reach" if s == t else "path_witness"
            return ReachCertificate(
                clean=True,
                pred=1,
                kind=kind,
                reason="checker_bfs_reconstructed_witness_validates",
                witness=reconstructed,
                reachable_by_checker=True,
            )
        # Double-check: existence without reconstruct (should agree)
        exists = reachable_bfs(n, edges, s, t)
        return ReachCertificate(
            clean=False,
            pred=1,
            kind="dirty_yes",
            reason=(
                "yes_pred_but_no_validating_path_witness"
                if not exists
                else "yes_pred_but_witness_reconstruction_failed"
            ),
            witness=None,
            reachable_by_checker=bool(exists),
        )

    # pred == 0 (NO)
    exists = reachable_bfs(n, edges, s, t)
    if not exists:
        return ReachCertificate(
            clean=True,
            pred=0,
            kind="unreachable_checker",
            reason="checker_bfs_confirms_unreachable",
            witness=None,
            reachable_by_checker=False,
        )
    return ReachCertificate(
        clean=False,
        pred=0,
        kind="dirty_no",
        reason="no_pred_but_checker_bfs_finds_path",
        witness=find_path_witness(n, edges, s, t),
        reachable_by_checker=True,
    )


def apply_cert_policy_preds(
    ens_preds: Any,
    rows: Sequence[dict[str, Any]],
    *,
    force_closed_dirty_yes: bool = True,
    force_open_dirty_no: bool = False,
) -> tuple[Any, list[ReachCertificate], list[int], list[int]]:
    """Apply fail-closed certificate policy to ens preds.

    Default: dirty YES → force pred=0; dirty NO → keep (no oracle open).
    Returns (policy_preds, certificates, dirty_yes_ids, dirty_no_ids).
    """
    import torch

    from reachability_gen.encode import parse_instance

    out = ens_preds.clone()
    certs: list[ReachCertificate] = []
    dirty_yes: list[int] = []
    dirty_no: list[int] = []
    for i, row in enumerate(rows):
        n, edges, s, t = parse_instance(str(row["encoding"]))
        n = int(row.get("n", n))
        s = int(row.get("s", s))
        t = int(row.get("t", t))
        pred = int(out[i].item())
        cert = certify_prediction(n, edges, s, t, pred)
        certs.append(cert)
        if not cert.clean and pred == 1:
            dirty_yes.append(i)
            if force_closed_dirty_yes:
                out[i] = 0
        elif not cert.clean and pred == 0:
            dirty_no.append(i)
            if force_open_dirty_no:
                out[i] = 1
    return out, certs, dirty_yes, dirty_no
