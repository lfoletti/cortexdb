package cortexdb

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	"github.com/liliang-cn/cortexdb/v2/pkg/graph"
)

// Entity nodes and collections.
//
// An entity node's id is its normalised name (graphEntityNodeID), so "Acme"
// in one collection and "Acme" in another are one node, with the mention
// edges and relations of both. That is what makes a glossary work across a
// whole brain, and what let a graph walk confined to one collection reach the
// other collection's chunks through the shared node. The node now records
// which collections mention it, in the property below, and the walk keeps to
// entities the searched collection knows (entityKnownIn).
//
// The list is a union that grows with every ingest; delete_document_graph
// detaches the entity from the document but does not shrink the list.
// upsert_entities has no collection of its own: it adds the collections of
// the stored chunks it names (mergeMentionCollections). An entity with no
// list — written before this property existed, or by upsert_entities with
// no stored chunk — is known everywhere.
// A caller scope reads the same list (scopedNode), and is stricter: there an
// entity with no list is known nowhere.

// entityCollectionsKey is the entity node property that lists the
// collections whose chunks mention the entity.
const entityCollectionsKey = "collections"

// mergeEntityCollections adds collection to each node's list, keeping what
// prior writes recorded. Mirrors mergeEntitySourceDocuments.
func mergeEntityCollections(nodes []*graph.GraphNode, prior map[string]*graph.GraphNode, collection string) {
	if collection == "" {
		return
	}
	for _, node := range nodes {
		if node.Properties == nil {
			node.Properties = map[string]interface{}{}
		}
		var before []string
		if was, ok := prior[node.ID]; ok && was.Properties != nil {
			before = toStringSlice(was.Properties[entityCollectionsKey])
		}
		node.Properties[entityCollectionsKey] = unionStrings(before, []string{collection})
	}
}

// mergeMentionCollections adds to each node's list the collections of the
// chunks that mention it, keeping what prior writes recorded. edges are the
// mention edges of the write, from chunk to entity, and chunkCollections the
// collection of each stored chunk among them. A chunk with no known
// collection adds nothing, and a node no such chunk mentions keeps its
// properties as they are. The list starts from the stored one, not from the
// node: a caller metadata key "collections", which can only be a string, is
// dropped when the write names a stored chunk.
func mergeMentionCollections(nodes []*graph.GraphNode, prior map[string]*graph.GraphNode, edges []*graph.GraphEdge, chunkCollections map[string]string) {
	found := make(map[string][]string)
	for _, edge := range edges {
		if collection := chunkCollections[edge.FromNodeID]; collection != "" {
			found[edge.ToNodeID] = unionStrings(found[edge.ToNodeID], []string{collection})
		}
	}
	for _, node := range nodes {
		if len(found[node.ID]) == 0 {
			continue
		}
		var before []string
		if was, ok := prior[node.ID]; ok {
			before = toStringSlice(was.Properties[entityCollectionsKey])
		}
		node.Properties[entityCollectionsKey] = unionStrings(before, found[node.ID])
	}
}

// mentionChunkCollections reads the collection of each stored chunk the
// edges start from, on embeddings.collection_id. A chunk the store does not
// hold, such as a stub, is not in the map.
func (db *DB) mentionChunkCollections(ctx context.Context, edges []*graph.GraphEdge) (map[string]string, error) {
	ids := make([]string, 0, len(edges))
	for _, edge := range edges {
		ids = append(ids, edge.FromNodeID)
	}
	out := make(map[string]string, len(ids))
	for _, batch := range stringChunks(ids, 1) {
		placeholders, args := sqlPlaceholders(batch)
		rows, err := db.query(ctx, `SELECT e.id, c.name FROM embeddings e
			JOIN collections c ON c.id = e.collection_id
			WHERE e.id IN (`+placeholders+`)`, args...)
		if err == nil {
			err = eachRow(rows, func() error {
				var id, name string
				err := rows.Scan(&id, &name)
				out[id] = name
				return err
			})
		}
		if err != nil {
			return nil, fmt.Errorf("load mention chunk collections: %w", err)
		}
	}
	return out, nil
}

// entityKnownIn reports whether an entity with these collections may be
// walked through by a search confined to collection. No confinement, or an
// entity with no list, is a yes.
func entityKnownIn(collections []string, collection string) bool {
	if collection == "" || len(collections) == 0 {
		return true
	}
	for _, c := range collections {
		if c == collection {
			return true
		}
	}
	return false
}

// scopedNode is node as a caller whose knowledge rule is permits may see it,
// and false when it may not see it at all. Chunk, document and memory nodes
// follow the rules of their documents and memories (hiddenAmong,
// hiddenMemoryNodes) and pass as they are, and so does every node when
// there is no knowledge rule. Any other node, an entity or a node a tool
// wrote such as a decision, is served only when a collection it lists is
// permitted, and then keeps its id and name: its description and its
// type are the last writer's, whichever collection that was, and the store
// does not record which, and its other properties, its vector and the
// times of its last write come from the same writes. The name is bounded
// by the id, which every writer derives from it. A node that lists no
// collection is not served, since nothing says which collection it came
// from.
func scopedNode(node *graph.GraphNode, permits func(collection string) bool) (*graph.GraphNode, bool) {
	if _, memory := memoryIDFromGraphNode(node.ID); permits == nil || memory || isDocumentGraphNode(node) {
		return node, true
	}
	if !slices.ContainsFunc(toStringSlice(node.Properties[entityCollectionsKey]), permits) {
		return nil, false
	}
	served := *node
	served.NodeType = ""
	served.Vector = nil
	served.UpdatedAt, served.ValidFrom, served.RecordedAt = time.Time{}, time.Time{}, time.Time{}
	served.Properties = map[string]any{}
	if name, ok := node.Properties["name"]; ok {
		served.Properties["name"] = name
	}
	return &served, true
}

// ofTypes leaves out of a walk's nodes, its seeds aside, those whose type
// is not one of nodeTypes, the edges that touch them and their scores. The
// walk filters on the type stored on each node, before the scope; an
// entity the scope reduces has no type (scopedNode), so it matches no type
// filter and is left out of nodes, edges and scores alike. Without a type
// filter, or for nodes the scope does not reduce, the walk's own filter
// has already decided and nothing changes.
func ofTypes(nodes []*graph.GraphNode, edges []*graph.GraphEdge, scores map[string]float64, nodeTypes, seeds []string) ([]*graph.GraphNode, []*graph.GraphEdge) {
	if len(nodeTypes) == 0 {
		return nodes, edges
	}
	kept := make(map[string]bool, len(nodes))
	nodes = slices.DeleteFunc(slices.Clone(nodes), func(node *graph.GraphNode) bool {
		keep := slices.Contains(seeds, node.ID) || slices.Contains(nodeTypes, node.NodeType)
		kept[node.ID] = keep
		if !keep {
			delete(scores, node.ID)
		}
		return !keep
	})
	edges = slices.DeleteFunc(slices.Clone(edges), func(edge *graph.GraphEdge) bool {
		return !kept[edge.FromNodeID] || !kept[edge.ToNodeID]
	})
	return nodes, edges
}

// isDocumentGraphNode reports whether a node is a chunk or document node of
// a document, which follows its document's rules rather than an entity's.
// It is decided by the id, which ingest derives from the document id
// (graphDocumentNodeID, graphChunkNodeID), and not by the node's type, which
// is the last writer's: an entity a writer typed "document" or "chunk" took
// the document branch and was served whole, with the description another
// collection wrote, to a caller confined to a collection that knows it.
func isDocumentGraphNode(node *graph.GraphNode) bool {
	docID, _ := node.Properties["document_id"].(string)
	return isDocumentNodeID(node.ID, docID)
}

// isDocumentNodeID reports whether id is the document node of docID or one
// of its chunk nodes.
func isDocumentNodeID(id, docID string) bool {
	if docID == "" {
		return false
	}
	return id == graphDocumentNodeID(docID) || strings.HasPrefix(id, "chunk:"+docID+":")
}
