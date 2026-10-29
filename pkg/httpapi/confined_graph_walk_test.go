package httpapi

import (
	"net/http"
	"strconv"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/authz"
)

// The REST twin of the gRPC test: a key confined to alpha asks the graph
// tools the same questions before and after the operator writes to beta, on
// one brain, and must get the same answers. POST /v1/query, whose graph lane
// a collection confines, is asked too.

type walkToolCall struct{ path, body string }

var confinedWalks = []walkToolCall{
	{"/v1/tools/search_chunks_by_entities", `{"entity_names":["Kestrel Vale"],"max_hops":3}`},
	{"/v1/tools/expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":1}`},
	{"/v1/tools/expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":2}`},
	{"/v1/tools/expand_graph", `{"node_ids":["entity:kestrel_vale"],"max_hops":1,"edge_types":["acquires"]}`},
	{"/v1/tools/knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":2}`},
	{"/v1/tools/knowledge_memory_neighbors", `{"entity_name":"Kestrel Vale","max_depth":1,"edge_types":["acquires"]}`},
	{"/v1/tools/knowledge_memory_expand_entity_context", `{"entity_names":["Kestrel Vale"],"max_hops":3}`},
	{"/v1/tools/search_paths", `{"entity_names":["Kestrel Vale","Heron Mill"],"max_depth":3,"include_bookkeeping":true}`},
	{"/v1/tools/knowledge_memory_shortest_path", `{"from_entity_name":"Kestrel Vale","to_entity_name":"Heron Mill"}`},
	{"/v1/query", `{"collection":"alpha","limit":10,"prefetch":[{"kind":"graph","entity_names":["Kestrel Vale"],"max_hops":3}]}`},
	{"/v1/tools/find_nodes", `{"names":["Kestrel Vale","Heron Mill","secret","deal"]}`},
}

// alphaAnswersBeforeAndAfter writes alpha's two documents, one mentioning
// Kestrel Vale and one Heron Mill, asks every walk as alpha, writes beta's
// document, and asks again.
func alphaAnswersBeforeAndAfter(t *testing.T, beta string) (before, after []string) {
	t.Helper()
	ks, err := authz.NewKeySet([]authz.Key{
		{ID: "operator", Secret: "op", Clearance: authz.ReadWrite},
		{ID: "alpha", Secret: "alpha-secret", Clearance: authz.ReadOnly, Scope: authz.Scope{Collection: "alpha"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	srv := newPolicyServer(t, Options{Keys: ks})
	write := func(body string) {
		wantStatus(t, srv, http.MethodPost, "/v1/tools/knowledge_save", "op", body, http.StatusOK)
	}
	ask := func() []string {
		answers := make([]string, 0, len(confinedWalks))
		for _, c := range confinedWalks {
			// "no path found" is an error status; the answer is the
			// status and the body together.
			status, payload := do(t, srv, http.MethodPost, c.path, "alpha-secret", c.body)
			answers = append(answers, strconv.Itoa(status)+" "+string(payload))
		}
		return answers
	}
	write(`{"knowledge_id":"a1","collection":"alpha","content":"kestrel vale runs the alpha mill.","entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:a1:000"]}]}`)
	write(`{"knowledge_id":"a2","collection":"alpha","content":"heron mill grinds alpha grain.","entities":[{"name":"Heron Mill","chunk_ids":["chunk:a2:000"]}]}`)
	before = ask()
	write(beta)
	return before, ask()
}

func TestAConfinedKeysGraphWalkDoesNotPassThroughAnotherCollectionsChunkOverHTTP(t *testing.T) {
	before, after := alphaAnswersBeforeAndAfter(t, `{"knowledge_id":"b1","collection":"beta","content":"kestrel vale is buying heron mill in secret.","entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:b1:000"]},{"name":"Heron Mill","chunk_ids":["chunk:b1:000"]}]}`)
	for i, c := range confinedWalks {
		if strings.Contains(after[i], "in secret") || strings.Contains(after[i], "chunk:b1") {
			t.Errorf("%s %s serves beta's chunk: %s", c.path, c.body, after[i])
		}
		if before[i] != after[i] {
			t.Errorf("%s %s: alpha's answer depends on beta's hidden co-occurrence\nbefore beta: %s\nafter beta:  %s", c.path, c.body, before[i], after[i])
		}
	}
}

func TestAConfinedKeysGraphWalkDoesNotFollowAnotherCollectionsRelationOverHTTP(t *testing.T) {
	before, after := alphaAnswersBeforeAndAfter(t, `{"knowledge_id":"b1","collection":"beta","content":"BETASECRET deal.","entities":[{"name":"Kestrel Vale","chunk_ids":["chunk:b1:000"]},{"name":"Heron Mill","chunk_ids":["chunk:b1:000"]}],"relations":[{"from":"Kestrel Vale","to":"Heron Mill","type":"acquires","chunk_ids":["chunk:b1:000"]}]}`)
	for i, c := range confinedWalks {
		if strings.Contains(after[i], "BETASECRET") || strings.Contains(after[i], "chunk:b1") {
			t.Errorf("%s %s serves beta's chunk: %s", c.path, c.body, after[i])
		}
		if heron := func(s string) bool { return strings.Contains(strings.ToLower(s), "heron mill") }; !heron(before[i]) && heron(after[i]) {
			t.Errorf("%s %s: alpha is served Heron Mill only because beta states Kestrel Vale acquires Heron Mill", c.path, c.body)
		}
		if before[i] != after[i] {
			t.Errorf("%s %s: alpha's answer depends on beta's relation\nbefore beta: %s\nafter beta:  %s", c.path, c.body, before[i], after[i])
		}
	}
}
