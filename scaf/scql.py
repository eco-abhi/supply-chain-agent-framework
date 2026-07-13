"""SCQL: compact, protocol-agnostic query notation (Section 3.5).

Design principles (following Memelang / SPL / ToON):
  - DEF an entity once per chain, reference by alias afterward
  - positional/delimiter fields instead of repeated JSON keys
  - identical payload grammar across HTTP / MCP / A2A

Includes a rough token counter so the benchmark can compare SCQL
against verbose JSON messaging.
"""
from __future__ import annotations

import json
import re

from .models import Entity


def rough_tokens(text: str) -> int:
    """Approximate LLM token count. Splits on whitespace and punctuation;
    close enough for relative comparisons between formats."""
    return len(re.findall(r"\w+|[^\w\s]", text))


class Chain:
    """One reasoning chain: registered entities + emitted messages.

    emit() is for messages within a single process/orchestrator, where
    self.entities is already shared memory. emit_to() is for messages
    that cross a process or agent boundary, where no shared memory can
    be assumed: it implements the self-describing header design flagged
    as unimplemented in the paper's first draft. Each participant gets
    the full entity definitions exactly once, the first time a chain
    reaches it; every message after that references entities by alias
    only, and a receiving side with no access to this Chain object can
    still decode it via parse_header().
    """

    def __init__(self, chain_id: str):
        self.chain_id = chain_id
        self.entities: dict[str, Entity] = {}
        self.messages: list[str] = []
        # per-participant memory of which aliases they've already been sent
        self.known_by: dict[str, set[str]] = {}

    # -- entity registration --
    def define(self, alias: str, entity_type: str, **fields) -> Entity:
        ent = Entity(alias=alias, entity_type=entity_type, fields=fields)
        self.entities[alias] = ent
        stmt = f"DEF {alias} = {entity_type}({','.join(f'{k}:{v}' for k, v in fields.items())})"
        self.messages.append(stmt)
        return ent

    # -- compact statements (single-process, shared memory assumed) --
    def emit(self, protocol: str, verb: str, target: str, payload: str = "") -> str:
        """e.g. emit('MCP','get','P1.stock') -> 'MCP.get P1.stock'"""
        stmt = f"{protocol}.{verb} {target}"
        if payload:
            stmt += f" {payload}"
        self.messages.append(stmt)
        return stmt

    # -- cross-process statements (self-describing header design) --
    def _referenced_aliases(self, *text_parts: str) -> list[str]:
        combined = " ".join(text_parts)
        # match whole-token alias references, avoid partial matches (e.g. P1 vs P10)
        return [a for a in self.entities if re.search(rf"\b{re.escape(a)}\b", combined)]

    def emit_to(self, participant: str, protocol: str, verb: str, target: str,
                payload: str = "") -> str:
        """Emit a message to a specific participant that may be running in
        a separate process. If that participant hasn't seen a referenced
        entity yet, its full definition is prepended as a header; once
        seen, later messages to that same participant reference it by
        alias alone. Returns the wire message actually sent, header
        included when present, so callers can inspect exactly what
        crossed the process boundary."""
        known = self.known_by.setdefault(participant, set())
        referenced = self._referenced_aliases(target, payload)
        new_aliases = [a for a in referenced if a not in known]

        stmt = f"{protocol}.{verb} {target}"
        if payload:
            stmt += f" {payload}"

        if new_aliases:
            header = " ".join(
                f"DEF {a}={self.entities[a].entity_type}"
                f"({','.join(f'{k}:{v}' for k, v in self.entities[a].fields.items())})"
                for a in new_aliases
            )
            known.update(new_aliases)
            wire_message = f"{header} || {stmt}"
        else:
            wire_message = stmt

        self.messages.append(wire_message)
        return wire_message

    # -- cost accounting --
    def scql_tokens(self) -> int:
        return sum(rough_tokens(m) for m in self.messages)

    def json_equivalent_tokens(self) -> int:
        """What the same conversation costs if every message re-serializes
        full entities as verbose JSON (the no-carry-forward baseline)."""
        total = 0
        for m in self.messages:
            # every message in the baseline carries all referenced entities in full
            referenced = [e for a, e in self.entities.items() if a in m]
            body = {"message": m}
            for e in referenced:
                body[e.alias] = json.loads(e.verbose())
            total += rough_tokens(json.dumps(body))
        return total


def parse_header(wire_message: str) -> tuple[dict[str, Entity], str]:
    """Decode a wire message produced by emit_to(), with no access to the
    sending Chain object, the way a genuinely separate process would.
    Returns (newly defined entities, the statement itself)."""
    if "||" in wire_message:
        header, stmt = wire_message.split("||", 1)
        header, stmt = header.strip(), stmt.strip()
    else:
        header, stmt = "", wire_message.strip()

    defined: dict[str, Entity] = {}
    for m in re.finditer(r"DEF (\w+)=(\w+)\(([^)]*)\)", header):
        alias, etype, fieldstr = m.groups()
        fields = {}
        if fieldstr:
            for pair in fieldstr.split(","):
                k, v = pair.split(":", 1)
                fields[k] = v
        defined[alias] = Entity(alias=alias, entity_type=etype, fields=fields)
    return defined, stmt


class RemoteParticipant:
    """Simulates a genuinely separate process: it starts knowing nothing
    and builds its own local entity cache purely by decoding incoming
    wire messages via parse_header(), never touching the sender's Chain
    object. Used to prove the header design round-trips correctly
    without shared memory, not just that emit_to() looks reasonable."""

    def __init__(self, name: str):
        self.name = name
        self.local_entities: dict[str, Entity] = {}
        self.received: list[str] = []

    def receive(self, wire_message: str) -> str:
        new_entities, stmt = parse_header(wire_message)
        self.local_entities.update(new_entities)
        self.received.append(stmt)
        return stmt

    def knows(self, alias: str) -> bool:
        return alias in self.local_entities
