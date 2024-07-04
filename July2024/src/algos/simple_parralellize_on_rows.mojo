from algorithm.functional import vectorize, parallelize

fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    
    @parameter
    fn calc_row(m: Int):
        for k in range(K):
            for n in range(N):
                res[m, n] += a[m, k] * b[k, n]

    parallelize[calc_row](M, M)

