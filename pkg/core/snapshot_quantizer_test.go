package core

import (
	"context"
	"fmt"
	"path/filepath"
	"testing"

	"github.com/liliang-cn/cortexdb/v2/pkg/quantization"
)

// quantizedAs opens a store as cortexdb.Open does: the dimension is known, so
// a quantizer exists before the first vector goes in.
func quantizedAs(q QuantizationConfig) func(*Config) {
	return func(c *Config) {
		c.VectorDim = 32
		c.Quantization = q
	}
}

var sq8Quantization = QuantizationConfig{Enabled: true, Type: "scalar", NBits: 8}

var snapshotQuantizations = []struct {
	name string
	q    QuantizationConfig
}{
	{"none", QuantizationConfig{}},
	{"off", DefaultQuantizationConfig()}, // off, but typed scalar 8-bit
	{"binary-off", QuantizationConfig{Type: "binary"}},
	{"sq8", sq8Quantization},
	{"sq4", QuantizationConfig{Enabled: true, Type: "scalar", NBits: 4}},
	{"sq0", QuantizationConfig{Enabled: true, Type: "scalar"}}, // no valid bits: no quantizer is made
	{"binary", QuantizationConfig{Enabled: true, Type: "binary"}},
}

// trainedIVF opens the store as IVF and trains it, so its close writes an IVF
// snapshot.
func trainedIVF(t *testing.T, path string) *SQLiteStore {
	t.Helper()
	store := openResourceStore(t, path, ivfStore)
	if err := store.TrainIndex(context.Background(), 4); err != nil {
		t.Fatal(err)
	}
	return store
}

// ageRows makes every row older than the snapshot, as in a store reopened a
// day later: reconciliation then has no recent row to rewrite into the index.
func ageRows(t *testing.T, path string) {
	t.Helper()
	store := openResourceStore(t, path, func(c *Config) { c.HNSW.Enabled = false; c.IndexType = IndexTypeFlat })
	if _, err := store.db.Exec(`UPDATE embeddings SET created_at = datetime('now', '-2 days')`); err != nil {
		t.Fatal(err)
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
}

// quantizes is whether q makes a quantizer: on, and binary or scalar with 1
// to 8 bits.
func quantizes(q QuantizationConfig) bool {
	return q.Enabled && (q.Type == "binary" || q.NBits >= 1 && q.NBits <= 8)
}

// holdsCodesOf checks the index holds what q asks for: codes of its kind and
// bits, or float vectors when q makes no quantizer.
func holdsCodesOf(t *testing.T, store *SQLiteStore, q QuantizationConfig) {
	t.Helper()
	switch quantizer := store.quantizer.(type) {
	case nil:
		if quantizes(q) {
			t.Fatal("quantization is on and the store has no quantizer")
		}
	case *quantization.BinaryQuantizer:
		if !q.Enabled || q.Type != "binary" {
			t.Fatalf("binary quantizer under %+v", q)
		}
	case *quantization.ScalarQuantizer: // any type but "binary" means scalar
		if !q.Enabled || q.Type == "binary" || quantizer.NBits != q.NBits {
			t.Fatalf("%d-bit scalar quantizer under %+v", quantizer.NBits, q)
		}
	}
	for id, node := range store.hnswIndex.Nodes {
		if quantizes(q) && (node.Vector != nil || node.Quantized == nil) {
			t.Fatalf("node %s holds no codes under %+v", id, q)
		}
		if !quantizes(q) && node.Vector == nil {
			t.Fatalf("node %s holds no float32 vector, and nothing reads its codes", id)
		}
	}
}

func selfFound(t *testing.T, store *SQLiteStore, vecs [][]float32) int {
	t.Helper()
	found := 0
	for i, v := range vecs {
		res, err := store.Search(context.Background(), v, SearchOptions{TopK: 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(res) == 1 && res[0].ID == fmt.Sprintf("v%03d", i) {
			found++
		}
	}
	return found
}

// A snapshot of codes can only be read by the quantizer that wrote them. A
// store reopened without quantization, or with another kind, used to panic
// (no quantization) or to load the codes and search them with nothing that
// could read them; it rebuilds its index from the table instead.
func TestCodeSnapshotIsReadOnlyByItsOwnQuantizer(t *testing.T) {
	for _, from := range snapshotQuantizations {
		for _, to := range snapshotQuantizations {
			t.Run(from.name+"->"+to.name, func(t *testing.T) {
				path := filepath.Join(t.TempDir(), "s.db")
				store := openResourceStore(t, path, quantizedAs(from.q))
				vecs := seedVectors(t, store, 120, 32)
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				ageRows(t, path)

				store = openResourceStore(t, path, quantizedAs(to.q))
				defer func() { _ = store.Close() }()
				rebuilt := store.snapshotChanges() != 0
				if from.name == to.name && rebuilt {
					t.Fatal("a snapshot reopened as it was written was rebuilt")
				}
				if quantizes(from.q) && from.name != to.name && !rebuilt {
					t.Fatal("a snapshot of codes was loaded under another quantization")
				}
				holdsCodesOf(t, store, to.q)
				// Scalar codes lose a few on random vectors even from their own snapshot.
				want := len(vecs)
				if to.q.Type != "binary" && quantizes(to.q) {
					want = len(vecs) * 3 / 4
				}
				if got := selfFound(t, store, vecs); got < want {
					t.Fatalf("a vector searched for by itself was found %d times in %d, want at least %d", got, len(vecs), want)
				}

				// The rebuild is paid once: the snapshot it wrote is the next one loaded.
				if err := store.Close(); err != nil {
					t.Fatal(err)
				}
				store = openResourceStore(t, path, quantizedAs(to.q))
				if store.snapshotChanges() != 0 {
					t.Fatal("the snapshot written after the reopen was not loaded on the next")
				}
				holdsCodesOf(t, store, to.q)
			})
		}
	}
}

// Once rebuilt without quantization, the store writes a float snapshot and
// drops the quantizer saved with the old one: the next open loads it.
func TestRebuildWithoutQuantizationDropsTheOldQuantizer(t *testing.T) {
	for _, off := range snapshotQuantizations[:2] {
		t.Run(off.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "s.db")
			store := openResourceStore(t, path, quantizedAs(sq8Quantization))
			seedVectors(t, store, 120, 32)
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			ageRows(t, path)
			store = openResourceStore(t, path, quantizedAs(off.q))
			if err := store.saveIndexSnapshot(context.Background()); err != nil || store.snapshotChanges() != 0 {
				t.Fatalf("saving the rebuilt index: %v, %d changes left", err, store.snapshotChanges())
			}
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}

			store = openResourceStore(t, path, quantizedAs(off.q))
			defer func() { _ = store.Close() }()
			if n := quantizerRows(t, store); n != 0 {
				t.Fatal("the quantizer of the old snapshot was kept beside a float one")
			}
			if store.snapshotChanges() != 0 {
				t.Fatal("the float snapshot written after the rebuild was not loaded")
			}
		})
	}
}

func quantizerRows(t *testing.T, store *SQLiteStore) int {
	t.Helper()
	var rows int
	if err := store.db.QueryRow(`SELECT COUNT(*) FROM index_snapshots WHERE type = 'QUANTIZER'`).Scan(&rows); err != nil {
		t.Fatal(err)
	}
	return rows
}

// The quantizer row changes with the snapshot or not at all. When it cannot
// be written, the new snapshot is not written either: the store keeps the old
// snapshot and its quantizer, a pair that still reads right under the old
// configuration and is rebuilt under the new one.
func TestASnapshotAndItsQuantizerAreWrittenTogether(t *testing.T) {
	sq4 := QuantizationConfig{Enabled: true, Type: "scalar", NBits: 4}
	for _, tc := range []struct {
		name, trigger string
		to            QuantizationConfig
	}{
		{"drop refused", `BEFORE DELETE ON index_snapshots WHEN old.type = 'QUANTIZER'`, QuantizationConfig{}},
		{"replace refused", `BEFORE INSERT ON index_snapshots WHEN new.type = 'QUANTIZER'`, sq4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "s.db")
			store := openResourceStore(t, path, quantizedAs(sq8Quantization))
			vecs := seedVectors(t, store, 120, 32)
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			ageRows(t, path)
			store = openResourceStore(t, path, quantizedAs(tc.to))
			if _, err := store.db.Exec(`CREATE TRIGGER keep_quantizer ` + tc.trigger + ` BEGIN SELECT RAISE(ABORT, 'kept'); END`); err != nil {
				t.Fatal(err)
			}
			_ = store.Close() // logs the refused snapshot and closes

			store = openResourceStore(t, path, quantizedAs(sq8Quantization))
			if store.snapshotChanges() != 0 {
				t.Fatal("the SQ8 snapshot was not kept")
			}
			holdsCodesOf(t, store, sq8Quantization)
			if got := selfFound(t, store, vecs); got < len(vecs)*3/4 {
				t.Fatalf("the kept SQ8 pair found a vector by itself %d times in %d", got, len(vecs))
			}
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}

			store = openResourceStore(t, path, quantizedAs(tc.to))
			defer func() { _ = store.Close() }()
			if store.snapshotChanges() == 0 {
				t.Fatal("the SQ8 snapshot was loaded under another configuration")
			}
			holdsCodesOf(t, store, tc.to)
			if got := selfFound(t, store, vecs); got < len(vecs)*3/4 {
				t.Fatalf("a vector searched for by itself was found %d times in %d", got, len(vecs))
			}
		})
	}
}

func ivfStore(c *Config) {
	c.VectorDim = 32
	c.Quantization = QuantizationConfig{} // as cortexdb.DefaultConfig leaves it
	c.HNSW.Enabled = false
	c.IndexType = IndexTypeIVF
}

// IVF keeps no codes: a quantizer saved by an earlier HNSW index is not its
// business. It used to panic the open; it must not make IVF rebuild either.
func TestIVFIgnoresAQuantizerSavedByHNSW(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.db")
	store := openResourceStore(t, path, quantizedAs(sq8Quantization))
	vecs := seedVectors(t, store, 120, 32)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = trainedIVF(t, path)
	if got := selfFound(t, store, vecs); got < len(vecs) {
		t.Fatalf("the IVF store found a vector by itself %d times in %d", got, len(vecs))
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	ageRows(t, path)
	store = openResourceStore(t, path, ivfStore)
	defer func() { _ = store.Close() }()
	if !store.ivfIndex.Trained || store.snapshotChanges() != 0 {
		t.Fatal("the trained IVF snapshot was not loaded")
	}
}

// The quantizer belongs to the HNSW snapshot: a spell as IVF keeps it, so
// the codes left in that snapshot are still known for what they are.
func TestIVFKeepsTheQuantizerOfTheHNSWSnapshot(t *testing.T) {
	path := filepath.Join(t.TempDir(), "s.db")
	store := openResourceStore(t, path, quantizedAs(sq8Quantization))
	vecs := seedVectors(t, store, 120, 32)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	store = trainedIVF(t, path)
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	ageRows(t, path)

	store = openResourceStore(t, path, quantizedAs(QuantizationConfig{}))
	defer func() { _ = store.Close() }()
	if got := selfFound(t, store, vecs); got < len(vecs) {
		t.Fatalf("a vector searched for by itself was found %d times in %d", got, len(vecs))
	}
}
