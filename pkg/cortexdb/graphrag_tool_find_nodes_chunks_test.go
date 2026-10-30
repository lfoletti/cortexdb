package cortexdb

import (
	"context"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// With skipChunks, the scan returns no chunk row: their text is not read,
// which is where the time of a call goes on a store of passages.
func TestFindNodesScanLeavesChunksUnread(t *testing.T) {
	db, _, ctx := findNodesTestDB(t)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "notes", Content: "Every Two-Phase Commit needs a coordinator."}); err != nil {
		t.Fatalf("save: %v", err)
	}
	for _, skip := range []bool{false, true} {
		rows, err := db.query(ctx, findNodesScan(skip))
		if err != nil {
			t.Fatalf("scan: %v", err)
		}
		chunks, others := 0, 0
		for rows.Next() {
			var id, content, nodeType string
			if err := rows.Scan(&id, &content, &nodeType); err != nil {
				t.Fatalf("row: %v", err)
			}
			if strings.HasPrefix(id, "chunk:") {
				chunks++
			} else {
				others++
			}
		}
		_ = rows.Close()
		if skip != (chunks == 0) || others == 0 {
			t.Fatalf("skipChunks=%v read %d chunk rows and %d others", skip, chunks, others)
		}
	}
}

// A chunk is a passage, not a name: one that mentions "Two-Phase Commit"
// matches it by containment. A caller resolving a name can leave chunk text
// out of the match, and the call no longer reads it; without the argument
// the tool answers as it always has.
func TestFindNodesSkipsChunkTextWhenAsked(t *testing.T) {
	db, tools, ctx := findNodesTestDB(t)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "notes", Content: "Every Two-Phase Commit needs a coordinator that outlives its participants."}); err != nil {
		t.Fatalf("save: %v", err)
	}
	chunks := func(resp *ToolFindNodesResponse) int {
		if len(resp.Matches) != 1 {
			t.Fatalf("want one match per name: %+v", resp.Matches)
		}
		n := 0
		for _, node := range resp.Matches[0].Nodes {
			if node.NodeType == "chunk" {
				n++
			}
		}
		return n
	}

	resp, err := tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Two-Phase Commit"}, Limit: 10})
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if chunks(resp) == 0 {
		t.Fatalf("without skip_chunks, the passage that mentions the name is a match, as before: %+v", resp.Matches[0].Nodes)
	}

	resp, err = tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Two-Phase Commit"}, Limit: 10, SkipChunks: true})
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if chunks(resp) != 0 {
		t.Fatalf("skip_chunks served a chunk: %+v", resp.Matches[0].Nodes)
	}
	if len(resp.Matches[0].Nodes) == 0 || resp.Matches[0].Nodes[0].Content != "Two-Phase Commit" || resp.Matches[0].Match != "exact" {
		t.Fatalf("skip_chunks lost the entity: %+v", resp.Matches[0])
	}

	out, err := tools.Call(ctx, "find_nodes", []byte(`{"names":["Two-Phase Commit"],"limit":10,"skip_chunks":true}`))
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if got := out.(*ToolFindNodesResponse); chunks(got) != 0 || len(got.Matches[0].Nodes) == 0 {
		t.Fatalf("skip_chunks did not travel through the tool call: %+v", got.Matches[0].Nodes)
	}
}

func TestFindNodesDeclaresSkipChunks(t *testing.T) {
	_, tools, _ := findNodesTestDB(t)
	for _, def := range tools.Definitions() {
		if def.Name != "find_nodes" {
			continue
		}
		props, _ := def.InputSchema["properties"].(map[string]any)
		prop, _ := props["skip_chunks"].(map[string]any)
		if prop["type"] != "boolean" {
			t.Fatalf("find_nodes does not declare skip_chunks as a boolean: %v", props["skip_chunks"])
		}
		return
	}
	t.Fatal("find_nodes is not in the tool definitions")
}

// skip_chunks knows a chunk by its id, not by the type its last writer gave
// it. Beta types alpha's entity "chunk": the entity is still matched, without
// a scope and under alpha's, so alpha cannot learn beta's type by asking with
// skip_chunks. The hidden-node filter, the ranking (exact first, then shorter
// matches) and the per-name limit hold as without the argument.
func TestFindNodesSkipChunksKnowsAChunkByItsID(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "find-nodes-skip.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	for _, save := range []KnowledgeSaveRequest{
		{KnowledgeID: "a1", Collection: "alpha", Content: "Kestrel Vale runs the alpha mill.",
			Entities: []ToolEntityInput{{Name: "Kestrel Vale", Type: "Person", ChunkIDs: []string{graphChunkNodeID("a1", 0)}}}},
		{KnowledgeID: "a3", Collection: "alpha", Content: "Kestrel Vale sings."},
		{KnowledgeID: "b1", Collection: "beta", Content: "Kestrel Vale works for the other side.",
			Entities: []ToolEntityInput{{Name: "Kestrel Vale", Type: "chunk", ChunkIDs: []string{graphChunkNodeID("b1", 0)}}}},
	} {
		if _, err := db.SaveKnowledge(ctx, save); err != nil {
			t.Fatal(err)
		}
	}
	find := func(ctx context.Context, limit int) []string {
		resp, err := db.GraphRAGTools().FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Kestrel Vale"}, Limit: limit, SkipChunks: true})
		if err != nil {
			t.Fatalf("find: %v", err)
		}
		if len(resp.Matches) != 1 {
			t.Fatalf("want one match per name: %+v", resp.Matches)
		}
		var ids []string
		for _, node := range resp.Matches[0].Nodes {
			ids = append(ids, node.ID)
		}
		return ids
	}
	kv := EntityNodeID("Kestrel Vale")
	all := find(ctx, 10)
	if len(all) == 0 || all[0] != kv {
		t.Fatalf("without a scope, the entity beta typed chunk is not the first match: %v", all)
	}
	alpha := WithCallerScope(ctx, alphaScope())
	want := []string{kv, graphDocumentNodeID("a3"), graphDocumentNodeID("a1")}
	if got := find(alpha, 10); !slices.Equal(got, want) {
		t.Fatalf("under alpha's scope: %v, want %v (the entity, then alpha's documents shortest first, nothing of beta's)", got, want)
	}
	if got := find(alpha, 1); !slices.Equal(got, want[:1]) {
		t.Fatalf("under alpha's scope with limit 1: %v, want %v", got, want[:1])
	}
}

// A batched call leaves the chunks out for every name, and a memory node, not
// a chunk, is still matched.
func TestFindNodesSkipChunksServesMemoriesAndEveryName(t *testing.T) {
	db, tools, ctx := findNodesTestDB(t)
	if _, err := db.SaveKnowledge(ctx, KnowledgeSaveRequest{KnowledgeID: "notes", Content: "Every Two-Phase Commit needs a coordinator; Paxos needs none."}); err != nil {
		t.Fatalf("save: %v", err)
	}
	if _, err := db.SaveMemory(ctx, MemorySaveRequest{MemoryID: "m1", Content: "Paxos needs a quorum of acceptors.", Scope: "global",
		Entities: []ToolEntityInput{{Name: "Paxos", Type: "Concept"}}}); err != nil {
		t.Fatalf("save memory: %v", err)
	}
	resp, err := tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Two-Phase Commit", "Paxos"}, Limit: 10, SkipChunks: true})
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if len(resp.Matches) != 2 {
		t.Fatalf("want one match per name: %+v", resp.Matches)
	}
	memory := false
	for _, match := range resp.Matches {
		if len(match.Nodes) == 0 {
			t.Fatalf("%q: no match", match.Name)
		}
		for _, node := range match.Nodes {
			if strings.HasPrefix(node.ID, "chunk:") {
				t.Fatalf("%q: skip_chunks served chunk %s", match.Name, node.ID)
			}
			memory = memory || node.ID == memoryGraphNodeID("m1")
		}
	}
	if !memory {
		t.Fatalf("the memory that mentions Paxos is not matched: %+v", resp.Matches[1].Nodes)
	}
}

// The rule is the id prefix "chunk:", colon included, and nothing else: a run
// whose id begins "chunk:" is left out, one named "chunky-run" is matched.
func TestFindNodesSkipChunksLeavesOutTheChunkPrefixOnly(t *testing.T) {
	db, tools, ctx := findNodesTestDB(t)
	for _, id := range []string{"chunk:r1", "chunky-run"} {
		if _, err := db.StartRun(ctx, RunStart{ID: id, Agent: "tester", Task: "Orchid Protocol drill"}); err != nil {
			t.Fatalf("start run %s: %v", id, err)
		}
	}
	resp, err := tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Orchid Protocol"}, Limit: 10, SkipChunks: true})
	if err != nil {
		t.Fatalf("find: %v", err)
	}
	if len(resp.Matches) != 1 {
		t.Fatalf("want one match per name: %+v", resp.Matches)
	}
	var ids []string
	for _, node := range resp.Matches[0].Nodes {
		ids = append(ids, node.ID)
	}
	if !slices.Equal(ids, []string{"chunky-run"}) {
		t.Fatalf("skip_chunks served %v, want [chunky-run]", ids)
	}
}
