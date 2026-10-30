package main

import (
	"bytes"
	"context"
	"log"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
)

func TestTheVectorIndexIsReadFromTheEnvironment(t *testing.T) {
	for _, tc := range []struct {
		index, oversample string
		name              string
		indexType         core.IndexType
		binary            int
	}{
		{"", "", "hnsw", core.IndexTypeHNSW, 0},
		{"hnsw", "", "hnsw", core.IndexTypeHNSW, 0},
		{"binary", "", "binary", core.IndexTypeBinary, 0},
		{"binary", "32", "binary (oversample 32)", core.IndexTypeBinary, 32},
		{"binary", "1000", "binary (oversample 1000)", core.IndexTypeBinary, 1000},
	} {
		t.Setenv("CORTEXDB_VECTOR_INDEX", tc.index)
		t.Setenv("CORTEXDB_BINARY_OVERSAMPLE", tc.oversample)
		config, name, err := vectorIndexFromEnv(cortexdb.DefaultConfig("x.db"))
		if err != nil {
			t.Fatalf("%q/%q: %v", tc.index, tc.oversample, err)
		}
		if name != tc.name || config.IndexType != tc.indexType || config.Quantization.Enabled || config.Binary.Oversample != tc.binary {
			t.Fatalf("%q/%q: name %q, index %v, quantization %+v, oversample %d", tc.index, tc.oversample, name, config.IndexType, config.Quantization, config.Binary.Oversample)
		}
		if config.Path != "x.db" {
			t.Fatalf("%q: the path was lost: %q", tc.index, config.Path)
		}
	}
}

func TestABadVectorIndexSettingStopsTheServer(t *testing.T) {
	for _, tc := range []struct{ index, oversample, says string }{
		{"hnsw8", "", "CORTEXDB_VECTOR_INDEX"},
		{"BINARY", "", "CORTEXDB_VECTOR_INDEX"},
		{"HNSW", "", "CORTEXDB_VECTOR_INDEX"},
		{"flat", "", "CORTEXDB_VECTOR_INDEX"},
		{"binary ", "", "CORTEXDB_VECTOR_INDEX"},
		{" hnsw", "", "CORTEXDB_VECTOR_INDEX"},
		{"hnsw-sq8", "", "CORTEXDB_VECTOR_INDEX"},
		{"binary", "many", "CORTEXDB_BINARY_OVERSAMPLE"},
		{"binary", "0", "CORTEXDB_BINARY_OVERSAMPLE"},
		{"binary", "-4", "CORTEXDB_BINARY_OVERSAMPLE"},
		{"binary", "1001", "CORTEXDB_BINARY_OVERSAMPLE"},
		{"binary", "9223372036854775807", "CORTEXDB_BINARY_OVERSAMPLE"},
		// An oversample the chosen index ignores is a setting that does
		// nothing: said, not dropped.
		{"", "32", "CORTEXDB_BINARY_OVERSAMPLE"},
		{"hnsw", "32", "CORTEXDB_BINARY_OVERSAMPLE"},
	} {
		t.Setenv("CORTEXDB_VECTOR_INDEX", tc.index)
		t.Setenv("CORTEXDB_BINARY_OVERSAMPLE", tc.oversample)
		if _, _, err := vectorIndexFromEnv(cortexdb.DefaultConfig("x.db")); err == nil || !strings.Contains(err.Error(), tc.says) {
			t.Fatalf("%q/%q: err %v, want one naming %s", tc.index, tc.oversample, err, tc.says)
		}
		path := filepath.Join(t.TempDir(), "brain.db")
		if db, err := openStore(path); err == nil {
			_ = db.Close()
			t.Fatalf("%q/%q: the store opened", tc.index, tc.oversample)
		}
		if _, err := os.Stat(path); !os.IsNotExist(err) {
			t.Fatalf("%q/%q: openStore created %s before refusing", tc.index, tc.oversample, path)
		}
	}
}

// On PostgreSQL the bit index sets hnsw.ef_search to top_k × oversample,
// which pgvector refuses past 1000: binary is offered for SQLite only. hnsw
// on a PostgreSQL DSN is unchanged.
func TestBinaryIsForASQLiteStore(t *testing.T) {
	t.Setenv("CORTEXDB_BINARY_OVERSAMPLE", "")
	// Every form Open takes for PostgreSQL; nothing listens on port 1, so a
	// refusal that came too late would fail on the connection instead.
	for _, dsn := range []string{
		"postgres://cortex@127.0.0.1:1/brain?sslmode=disable",
		"postgresql://cortex@127.0.0.1:1/brain?sslmode=disable",
		"host=127.0.0.1 port=1 dbname=brain sslmode=disable",
	} {
		t.Setenv("CORTEXDB_VECTOR_INDEX", "binary")
		if _, err := openStore(dsn); err == nil || !strings.Contains(err.Error(), "SQLite") {
			t.Fatalf("binary on %q: err %v, want a refusal naming SQLite", dsn, err)
		}
		t.Setenv("CORTEXDB_VECTOR_INDEX", "hnsw")
		if config, _, err := vectorIndexFromEnv(cortexdb.DefaultConfig(dsn)); err != nil || config.IndexType != core.IndexTypeHNSW {
			t.Fatalf("hnsw on %q: %v, %v", dsn, config.IndexType, err)
		}
	}
}

// The store the server opens searches with the index the environment names.
// HNSW over 300 vectors misses no true nearest neighbour. A binary index
// misses fewer the longer its shortlist: more at oversample 1 than at the
// default (8), and none when the shortlist is the whole table. So the misses
// show which index answered, and that the oversample reached it.
func TestTheServerSearchesWithTheIndexItNames(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	vector := func() []float32 {
		v := make([]float32, 16)
		for i := range v {
			v[i] = float32(rng.NormFloat64())
		}
		return v
	}
	stored := make(map[string][]float32, 300)
	for i := 0; i < 300; i++ {
		stored[string(rune('a'+i/26))+string(rune('a'+i%26))] = vector()
	}
	queries := make([][]float32, 60)
	for i := range queries {
		queries[i] = vector()
	}
	cosine := func(a, b []float32) float64 {
		var dot, na, nb float64
		for i := range a {
			dot += float64(a[i] * b[i])
			na += float64(a[i] * a[i])
			nb += float64(b[i] * b[i])
		}
		return dot / math.Sqrt(na*nb)
	}
	nearest := func(q []float32) string {
		best, id := -2.0, ""
		for k, v := range stored {
			if c := cosine(q, v); c > best {
				best, id = c, k
			}
		}
		return id
	}

	ids := make([]string, 0, len(stored))
	for id := range stored {
		ids = append(ids, id)
	}
	slices.Sort(ids)
	misses := map[string]int{}
	for _, tc := range []struct{ index, oversample, logs, info string }{
		{"", "", "vector index: hnsw", "HNSW"},
		{"binary", "1", "vector index: binary (oversample 1)", "Binary"},
		{"binary", "", "vector index: binary", "Binary"},
		{"binary", "300", "vector index: binary (oversample 300)", "Binary"},
	} {
		t.Setenv("CORTEXDB_VECTOR_INDEX", tc.index)
		t.Setenv("CORTEXDB_BINARY_OVERSAMPLE", tc.oversample)
		var logged bytes.Buffer
		previous := log.Writer()
		log.SetOutput(&logged)
		db, err := openStore(filepath.Join(t.TempDir(), "brain.db"))
		log.SetOutput(previous)
		if err != nil {
			t.Fatalf("%q/%q: open: %v", tc.index, tc.oversample, err)
		}
		if !strings.Contains(logged.String(), tc.logs) {
			t.Fatalf("%q/%q: logged %q, want %q", tc.index, tc.oversample, logged.String(), tc.logs)
		}
		if got := db.Info().IndexType; got != tc.info {
			t.Fatalf("%q/%q: index %q, want %q", tc.index, tc.oversample, got, tc.info)
		}
		ctx := context.Background()
		for _, id := range ids {
			if err := db.Vector().Upsert(ctx, &core.Embedding{ID: id, Vector: stored[id], Content: id}); err != nil {
				t.Fatalf("upsert %s: %v", id, err)
			}
		}
		key := tc.index + "/" + tc.oversample
		for _, q := range queries {
			got, err := db.Vector().Search(ctx, q, core.SearchOptions{TopK: 1})
			if err != nil || len(got) != 1 {
				t.Fatalf("%q/%q: search: %+v, %v", tc.index, tc.oversample, got, err)
			}
			if got[0].ID != nearest(q) {
				misses[key]++
			}
		}
		if err := db.Close(); err != nil {
			t.Fatalf("close: %v", err)
		}
	}
	if misses["/"] != 0 || misses["binary/300"] != 0 || misses["binary/"] == 0 || misses["binary/1"] <= misses["binary/"] {
		t.Fatalf("nearest missed out of %d: %v; want none for hnsw and the whole-table shortlist, and more at oversample 1 than at the default, which misses some", len(queries), misses)
	}
}
