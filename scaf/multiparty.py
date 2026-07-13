"""5. Multi-party SCQL header-caching benchmark.

Compares three regimes for a chain that fans out messages to several
participants (agents in separate processes):
  a. naive JSON, every message to every participant carries full
     verbose JSON for every entity it references (no caching at all)
  b. SCQL, no header caching, every message re-sends the DEF block
     (compact syntax, but same "no memory" assumption as (a))
  c. SCQL with the self-describing header design (emit_to), each
     participant gets each entity's definition exactly once
"""
from __future__ import annotations

import random

from .scql import Chain, RemoteParticipant, rough_tokens


def multiparty_scql_benchmark(n_participants: int = 5, n_messages: int = 200,
                              seed: int = 1):
    rng = random.Random(seed)
    participants = [f"agent{i}" for i in range(n_participants)]

    ch = Chain("multiparty")
    ch.define("P1", "Part", id=4471, name="BatteryModuleHousing")
    ch.define("S1", "Supplier", id="ABC001", name="ABCCorp")
    entity_aliases = ["P1", "S1"]

    remotes = {p: RemoteParticipant(p) for p in participants}
    ever_sent: dict[str, set[str]] = {p: set() for p in participants}

    naive_json_tokens = 0
    naive_scql_tokens = 0   # SCQL syntax but re-sent every message, no caching
    header_scql_tokens = 0

    for _ in range(n_messages):
        participant = rng.choice(participants)
        alias = rng.choice(entity_aliases)
        verb_payload = f"check {alias}.status"
        ever_sent[participant].add(alias)

        # (c) header-once design, actually round-tripped through a
        # separate RemoteParticipant with no shared memory
        wire = ch.emit_to(participant, "A2A", "ask", verb_payload)
        remotes[participant].receive(wire)
        header_scql_tokens += rough_tokens(wire)

        # (b) same compact grammar, but resend full DEF every time
        # (what SCQL would cost with no per-participant memory at all)
        ent = ch.entities[alias]
        redef = f"DEF {alias}={ent.entity_type}({','.join(f'{k}:{v}' for k, v in ent.fields.items())})"
        naive_scql_tokens += rough_tokens(f"{redef} || A2A.ask {verb_payload}")

        # (a) verbose JSON every message, every entity referenced, in full
        import json
        body = {"message": f"A2A.ask {verb_payload}", alias: json.loads(ent.verbose())}
        naive_json_tokens += rough_tokens(json.dumps(body))

    # correctness check: every participant should know, purely from
    # decoding wire messages with no shared memory, every entity it was
    # ever sent
    round_trip_verified = all(
        remotes[p].knows(a) for p in participants for a in ever_sent[p]
    )

    return {
        "n_participants": n_participants,
        "n_messages": n_messages,
        "naive_json_tokens": naive_json_tokens,
        "naive_scql_tokens": naive_scql_tokens,
        "header_scql_tokens": header_scql_tokens,
        "json_vs_header_ratio": naive_json_tokens / header_scql_tokens,
        "scql_naive_vs_header_ratio": naive_scql_tokens / header_scql_tokens,
        "round_trip_verified": round_trip_verified,
    }


if __name__ == "__main__":
    r = multiparty_scql_benchmark()
    print(f"Participants: {r['n_participants']}, messages: {r['n_messages']}")
    print(f"  Verbose JSON, no caching:      {r['naive_json_tokens']:,} tokens")
    print(f"  SCQL syntax, no caching:        {r['naive_scql_tokens']:,} tokens")
    print(f"  SCQL, header-once-per-agent:    {r['header_scql_tokens']:,} tokens")
    print(f"  Reduction vs verbose JSON:      {r['json_vs_header_ratio']:.1f}x")
    print(f"  Reduction vs uncached SCQL:     {r['scql_naive_vs_header_ratio']:.1f}x")
    print(f"  Round-trip decoding verified:   {r['round_trip_verified']}")
