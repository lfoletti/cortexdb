package main

import (
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/liliang-cn/cortexdb/v2/pkg/core"
	"github.com/liliang-cn/cortexdb/v2/pkg/cortexdb"
	"github.com/liliang-cn/cortexdb/v2/pkg/sqldialect"
)

// vectorIndexFromEnv sets the vector index of config from
// CORTEXDB_VECTOR_INDEX, and returns the name the startup log prints.
//
//   - hnsw (the default): float32 vectors in the graph, as before.
//   - binary: a 1-bit code per dimension in memory, 1/32 of the vectors,
//     each search rescored exactly from the stored vectors; no snapshot, the
//     codes are rebuilt from the embeddings table at open. SQLite only: on
//     PostgreSQL the bit index sets hnsw.ef_search to top_k × oversample,
//     which pgvector refuses past 1000, so the server does not offer it there.
//
// CORTEXDB_BINARY_OVERSAMPLE is the binary index's shortlist multiplier
// (BinaryConfig.Oversample). Recall is lower at small dimensions, so a
// deployment can trade a little search time for it. It is at most 1000,
// which keeps top_k × oversample from overflowing. Set with another index, it
// would do nothing, and is refused rather than ignored, like any value this
// function does not know.
func vectorIndexFromEnv(config cortexdb.Config) (cortexdb.Config, string, error) {
	index := os.Getenv("CORTEXDB_VECTOR_INDEX")
	oversample := os.Getenv("CORTEXDB_BINARY_OVERSAMPLE")
	if oversample != "" && index != "binary" {
		return config, "", fmt.Errorf("CORTEXDB_BINARY_OVERSAMPLE applies to CORTEXDB_VECTOR_INDEX=binary only")
	}
	switch index {
	case "", "hnsw":
		return config, "hnsw", nil
	case "binary":
		if sqldialect.KindForDSN(config.Path) == sqldialect.Postgres {
			return config, "", fmt.Errorf("CORTEXDB_VECTOR_INDEX=binary is for a SQLite store, not a PostgreSQL one")
		}
		config.IndexType = core.IndexTypeBinary
		if oversample == "" {
			return config, "binary", nil
		}
		n, err := strconv.Atoi(oversample)
		if err != nil || n < 1 || n > 1000 {
			return config, "", fmt.Errorf("invalid CORTEXDB_BINARY_OVERSAMPLE %q: want a whole number from 1 to 1000", oversample)
		}
		config.Binary.Oversample = n
		return config, fmt.Sprintf("binary (oversample %d)", n), nil
	}
	return config, "", fmt.Errorf("invalid CORTEXDB_VECTOR_INDEX %q: want hnsw or binary", index)
}

// openStore opens the store at path with the vector index the environment
// names, and logs that index, as main logs the embedder.
func openStore(path string, opts ...cortexdb.Option) (*cortexdb.DB, error) {
	config, vectorIndex, err := vectorIndexFromEnv(cortexdb.DefaultConfig(path))
	if err != nil {
		return nil, err
	}
	log.Printf("vector index: %s", vectorIndex)
	return cortexdb.Open(config, opts...)
}
