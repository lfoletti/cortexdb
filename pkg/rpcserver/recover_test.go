package rpcserver

import (
	"context"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// A handler that panics is answered with Internal; the process lives on.
func TestAPanickingHandlerIsAnsweredWithInternal(t *testing.T) {
	info := &grpc.UnaryServerInfo{FullMethod: "/cortexdb.v1.ToolsService/CallTool"}
	resp, err := recoverInterceptor()(context.Background(), nil, info, func(context.Context, any) (any, error) {
		_ = make([]int, 0, 1<<62)
		return "unreached", nil
	})
	if resp != nil || status.Code(err) != codes.Internal {
		t.Fatalf("got %v, %v; want nil, Internal", resp, err)
	}
	resp, err = recoverInterceptor()(context.Background(), nil, info, func(context.Context, any) (any, error) { return "ok", nil })
	if resp != "ok" || err != nil {
		t.Fatalf("a handler that returns is passed through: %v, %v", resp, err)
	}
}
