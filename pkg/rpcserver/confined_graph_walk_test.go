package rpcserver

import (
	"strings"
	"testing"

	"google.golang.org/grpc/status"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
	rpcv1 "github.com/liliang-cn/cortexdb/v2/pkg/rpc/v1"
)

// A key confined to one collection asks the graph tools the same questions
// before and after the operator writes to another collection, on one brain.
// The graph tools must not walk through what the other collection wrote,
// or the answer changes and tells the key what that collection holds, even
// when every node it serves is the key's own.

// walkCallerKeys is the operator and a reader confined to alpha.
func walkCallerKeys(t *testing.T) *authz.KeySet {
	t.Helper()
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op-secret", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	return ks
}

type walkToolCall struct{ tool, args string }

// alphaKnowsBoth is alpha's half of the brain: one document mentions Kestrel
// Vale, the other Heron Mill, and nothing in alpha links the two.
var alphaKnowsBoth = []walkToolCall{
	{"knowledge_save", `{"knowledge_id":"a1","collection":"alpha","content":"Kestrel Vale runs the alpha mill.",
		"entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:a1:000"]}]}`},
	{"knowledge_save", `{"knowledge_id":"a2","collection":"alpha","content":"Heron Mill grinds alpha grain.",
		"entities":[{"name":"Heron Mill","chunk_ids":["chunk:a2:000"]}]}`},
}

// betaMentionsBoth: one chunk of beta mentions both entities alpha knows.
var betaMentionsBoth = walkToolCall{"knowledge_save", `{"knowledge_id":"b1","collection":"beta","content":"Kestrel Vale is buying Heron Mill in secret.",
	"entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:b1:000"]},{"name":"Heron Mill","chunk_ids":["chunk:b1:000"]}]}`}

// betaStatesARelation: beta states Kestrel Vale acquires Heron Mill.
var betaStatesARelation = walkToolCall{"knowledge_save", `{"knowledge_id":"b1","collection":"beta","content":"BETASECRET deal.",
	"entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:b1:000"]},{"name":"Heron Mill","chunk_ids":["chunk:b1:000"]}],
	"relations":[{"from":"Kestrel Vale","to":"Heron Mill","type":"acquires","chunk_ids":["chunk:b1:000"]}]}`}

// confinedWalks is each graph tool that walks, as the alpha key sends it.
var confinedWalks = []walkToolCall{
	{"search_chunks_by_entities", `{"entity_names":["Kestrel Vale"],"max_hops":3}`},
	{"expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":1}`},
	{"expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":2}`},
	{"expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":1,"edge_types":["acquires"]}`},
	{"expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":3,"relation_policies":{"mentions":{"weight":1}}}`},
	{"knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":1}`},
	{"knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":2}`},
	{"knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":1,"edge_types":["acquires"]}`},
	{"knowledge_memory_expand_entity_context", `{"entity_names":["Kestrel Vale"],"max_hops":3}`},
	{"search_paths", `{"entity_names":["Kestrel Vale","Heron Mill"],"max_depth":3}`},
	{"search_paths", `{"entity_names":["Kestrel Vale","Heron Mill"],"max_depth":3,"include_bookkeeping":true,"with_text":true}`},
	{"knowledge_memory_shortest_path", `{"from_entity_name":"Kestrel Vale","to_entity_name":"Heron Mill"}`},
	{"find_nodes", `{"names":["Kestrel Vale","Heron Mill","secret","deal"]}`},
}

// alphaAnswersBeforeAndAfter writes alpha's half as the operator, asks
// every walk as alpha, writes beta's document, and asks again.
func alphaAnswersBeforeAndAfter(t *testing.T, beta walkToolCall) (before, after []string) {
	t.Helper()
	tools := rpcv1.NewToolsServiceClient(newPolicyConn(t, Options{Keys: walkCallerKeys(t)}))
	write := func(w walkToolCall) {
		if _, err := tools.CallTool(asKey("op-secret"), &rpcv1.CallToolRequest{Name: w.tool, ArgsJson: w.args}); err != nil {
			t.Fatalf("operator %s: %v", w.tool, err)
		}
	}
	ask := func() []string {
		answers := make([]string, 0, len(confinedWalks))
		for _, c := range confinedWalks {
			resp, err := tools.CallTool(asKey("alpha-secret"), &rpcv1.CallToolRequest{Name: c.tool, ArgsJson: c.args})
			answers = append(answers, resp.GetResultJson()+" | "+status.Code(err).String()+" "+status.Convert(err).Message())
		}
		return answers
	}
	for _, w := range alphaKnowsBoth {
		write(w)
	}
	before = ask()
	write(beta)
	return before, ask()
}

func TestAConfinedKeysGraphWalkDoesNotPassThroughAnotherCollectionsChunk(t *testing.T) {
	before, after := alphaAnswersBeforeAndAfter(t, betaMentionsBoth)
	for i, c := range confinedWalks {
		if strings.Contains(after[i], "in secret") || strings.Contains(after[i], "chunk:b1") {
			t.Errorf("%s %s serves beta's chunk: %s", c.tool, c.args, after[i])
		}
		if before[i] != after[i] {
			t.Errorf("%s %s: alpha's answer depends on beta's hidden co-occurrence\nbefore beta: %s\nafter beta:  %s", c.tool, c.args, before[i], after[i])
		}
	}
}

func TestAConfinedKeysGraphWalkDoesNotFollowAnotherCollectionsRelation(t *testing.T) {
	before, after := alphaAnswersBeforeAndAfter(t, betaStatesARelation)
	for i, c := range confinedWalks {
		if strings.Contains(after[i], "BETASECRET") || strings.Contains(after[i], "chunk:b1") {
			t.Errorf("%s %s serves beta's chunk: %s", c.tool, c.args, after[i])
		}
		if !strings.Contains(before[i], "entity:heron_mill") && strings.Contains(after[i], "entity:heron_mill") {
			t.Errorf("%s %s: alpha is served Heron Mill only because beta states Kestrel Vale acquires Heron Mill", c.tool, c.args)
		}
		if before[i] != after[i] {
			t.Errorf("%s %s: alpha's answer depends on beta's relation\nbefore beta: %s\nafter beta:  %s", c.tool, c.args, before[i], after[i])
		}
	}
}
