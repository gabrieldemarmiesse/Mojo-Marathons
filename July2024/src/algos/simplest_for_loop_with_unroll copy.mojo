@always_inline
fn loop_over_M[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    @parameter
    if M < 8:
        @parameter
        for m in range(M):
            loop_over_K(res, a, b, m)
    else:
        for m in range(M):
            loop_over_K(res, a, b, m)

@always_inline
fn loop_over_K[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N], m: Int):
    @parameter
    if K < 8:
        @parameter
        for k in range(K):
            loop_over_N(res, a, b, m, k)
    else:
        for k in range(K):
            loop_over_N(res, a, b, m, k)

@always_inline
fn loop_over_N[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N], m: Int, k: Int):
    @parameter
    if N < 8:
        @parameter
        for n in range(N):
            res[m, n] += a[m, k] * b[k, n]
    else:
        for n in range(N):
            res[m, n] += a[m, k] * b[k, n]

fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):

    loop_over_M(res, a, b)
