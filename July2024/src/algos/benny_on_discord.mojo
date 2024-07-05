from algorithm.functional import vectorize, parallelize
from algorithm import parallel_memcpy



fn matmul[Type: DType, M: Int, N: Int, K: Int, //](inout res: Matrix[Type, M, N], a: Matrix[Type, M, K], b: Matrix[Type, K, N]):
    alias BLOCK_N = 1 # L1 / sizeof[Type]
    alias BLOCK_M = 1 # L2 / sizeof[Type]

    fn calculate_block[BM: Int, BN: Int](bm: Int, bn: Int) capturing:
        var acc = stack_allocation[BM * BN, Type]()
        memset_zero(acc, BM * BN)

        for k in range(K):
            var b = b.data + k * N

            for m in range(BM):
                var a_val = a[bm + m, k]
                var acc = acc + m * BN

                fn inner_n[W: Int](n: Int) capturing:
                    SIMD[size=W].store(acc, n, SIMD[size=W].load(b, bn + n).fma(a_val, SIMD[size=W].load(acc, n)))

                vectorize[inner_n, simdwidthof[Type](), size=BN]()

        for m in range(BM):
            parallel_memcpy(res.data + (bm + m) * N + bn, acc + m * BN, BN)

    fn process_block[BM: Int](bm: Int) capturing:
        @parameter
        for bn in range(0, N if BM else 0, BLOCK_N):
            calculate_block[BM, min(BLOCK_N, N - bn)](bm * BM if BM == BLOCK_M else bm, bn)

    parallelize[process_block[BLOCK_M]](M // BLOCK_M)
    process_block[M % BLOCK_M](M - M % BLOCK_M)
