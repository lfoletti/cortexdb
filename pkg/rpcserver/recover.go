package rpcserver

import (
	"context"
	"log"
	"runtime/debug"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// recoverInterceptor answers a handler that panics with Internal, and logs
// the panic with its stack. grpc-go does not recover a panicking handler, so
// one call that reached a bug, such as a limit no slice can hold, took the
// whole server down with it, every caller's connection included. It sits
// last in the chain, around the handler, so the interceptors before it
// (metrics, the request log, tracing, the policy) see an ordinary error and
// record it.
func recoverInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (resp any, err error) {
		defer func() {
			if p := recover(); p != nil {
				log.Printf("cortexdb: %s panicked: %v\n%s", info.FullMethod, p, debug.Stack())
				resp, err = nil, status.Error(codes.Internal, "internal error")
			}
		}()
		return handler(ctx, req)
	}
}
