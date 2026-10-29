package cortexdb

import (
	"context"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"
)

// A node takes the document branch of the caller scope by its id, which
// ingest derives from its document, never by its type, which the last
// writer chose: an entity another collection typed "document" or "chunk" is
// served to alpha as an entity, reduced to its id and name, not whole with
// that collection's description and collections list.
func TestUnderACallerScopeAnEntityTypedDocumentIsServedAsAnEntity(t *testing.T) {
	for _, typed := range []string{"document", "chunk"} {
		t.Run(typed, func(t *testing.T) {
			db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "typed.db")))
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = db.Close() })
			ctx := context.Background()
			for _, doc := range []KnowledgeSaveRequest{
				{KnowledgeID: "a1", Collection: "alpha", Content: "Kestrel Vale runs the mill.",
					Entities: []ToolEntityInput{{Name: "Kestrel Vale"}}},
				{KnowledgeID: "b1", Collection: "beta", Content: "Kestrel Vale, privately.",
					Entities: []ToolEntityInput{{Name: "Kestrel Vale", Type: typed, Description: "BETASECRETDESC"}}},
			} {
				if _, err := db.SaveKnowledge(ctx, doc); err != nil {
					t.Fatal(err)
				}
			}
			out, err := db.GraphRAGTools().GetNodes(WithCallerScope(ctx, alphaScope()), ToolGetNodesRequest{NodeIDs: []string{EntityNodeID("Kestrel Vale")}})
			if err != nil {
				t.Fatal(err)
			}
			raw, _ := json.Marshal(out)
			if strings.Contains(string(raw), "BETASECRETDESC") || strings.Contains(string(raw), "beta") || strings.Contains(string(raw), `"`+typed+`"`) {
				t.Fatalf("alpha is served beta's writing of the entity: %s", raw)
			}
			if !strings.Contains(string(raw), "Kestrel Vale") {
				t.Fatalf("alpha is not served its own entity: %s", raw)
			}
		})
	}
}

// find_nodes sizes its matches by the names it can match, not by the
// caller's limit: a limit past what a slice can hold answered with a panic,
// which takes cortexdb-grpc down, since its handlers do not recover.
func TestFindNodesTakesALimitPastAnySliceSize(t *testing.T) {
	_, tools, ctx := findNodesTestDB(t)
	if _, err := tools.FindNodes(ctx, ToolFindNodesRequest{Names: []string{"Limit"}, Limit: 1 << 62}); err != nil {
		t.Fatalf("find: %v", err)
	}
}

// Under a caller scope, find_nodes holds the same: asking for the type
// another collection gave an entity serves nothing of it, and a limit past
// any slice size answers.
func TestUnderACallerScopeFindNodesNeitherServesATypedEntityNorPanics(t *testing.T) {
	db, err := Open(DefaultConfig(filepath.Join(t.TempDir(), "typed-find.db")))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = db.Close() })
	ctx := context.Background()
	for _, doc := range []KnowledgeSaveRequest{
		{KnowledgeID: "a1", Collection: "alpha", Content: "Kestrel Vale runs the mill.", Entities: []ToolEntityInput{{Name: "Kestrel Vale"}}},
		{KnowledgeID: "b1", Collection: "beta", Content: "Kestrel Vale, privately.", Entities: []ToolEntityInput{{Name: "Kestrel Vale", Type: "document", Description: "BETASECRETDESC"}}},
	} {
		if _, err := db.SaveKnowledge(ctx, doc); err != nil {
			t.Fatal(err)
		}
	}
	scoped := WithCallerScope(ctx, alphaScope())
	for _, req := range []ToolFindNodesRequest{
		{Names: []string{"Kestrel Vale"}, NodeTypes: []string{"document"}},
		{Names: []string{"Kestrel Vale"}, Limit: 1 << 62},
	} {
		out, err := db.GraphRAGTools().FindNodes(scoped, req)
		if err != nil {
			t.Fatalf("%+v: %v", req, err)
		}
		raw, _ := json.Marshal(out)
		if strings.Contains(string(raw), "BETASECRETDESC") || strings.Contains(string(raw), `"beta"`) {
			t.Errorf("%+v serves beta's writing: %s", req, raw)
		}
		if len(req.NodeTypes) > 0 && strings.Contains(string(raw), "entity:kestrel_vale") {
			t.Errorf("node_types [document] serves the entity beta typed document: %s", raw)
		}
	}
}
