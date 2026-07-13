from scaf.multiparty import multiparty_scql_benchmark
from scaf.scql import Chain, RemoteParticipant, parse_header


def test_header_sent_once_per_participant():
    ch = Chain("t")
    ch.define("P1", "Part", id=4471, name="Widget")
    m1 = ch.emit_to("agentA", "MCP", "get", "P1.stock")
    m2 = ch.emit_to("agentA", "MCP", "get", "P1.status")
    assert "DEF P1" in m1
    assert "DEF P1" not in m2  # already known to agentA


def test_header_sent_separately_per_participant():
    ch = Chain("t")
    ch.define("P1", "Part", id=4471, name="Widget")
    m_a = ch.emit_to("agentA", "MCP", "get", "P1.stock")
    m_b = ch.emit_to("agentB", "MCP", "get", "P1.stock")
    assert "DEF P1" in m_a
    assert "DEF P1" in m_b  # agentB hasn't seen it yet, even though agentA has


def test_remote_participant_decodes_with_no_shared_memory():
    ch = Chain("t")
    ch.define("P1", "Part", id=4471, name="Widget")
    ch.define("S1", "Supplier", id="X1", name="Acme")
    remote = RemoteParticipant("finance")
    assert not remote.knows("P1")
    wire = ch.emit_to("finance", "A2A", "ask", "approve P1 via S1")
    remote.receive(wire)
    assert remote.knows("P1") and remote.knows("S1")
    assert remote.local_entities["P1"].fields["name"] == "Widget"


def test_parse_header_roundtrip_without_prior_definitions():
    # decode a wire message cold, simulating a brand-new process
    entities, stmt = parse_header("DEF P1=Part(id:4471,name:Widget) || MCP.get P1.stock")
    assert entities["P1"].entity_type == "Part"
    assert entities["P1"].fields["id"] == "4471"
    assert stmt == "MCP.get P1.stock"


def test_parse_header_no_header_present():
    entities, stmt = parse_header("MCP.get P1.stock")
    assert entities == {}
    assert stmt == "MCP.get P1.stock"


def test_multiparty_benchmark_round_trip_and_savings():
    result = multiparty_scql_benchmark(n_participants=5, n_messages=200)
    assert result["round_trip_verified"] is True
    # header caching should beat both baselines by a real margin
    assert result["json_vs_header_ratio"] > 3
    assert result["scql_naive_vs_header_ratio"] > 1.5
