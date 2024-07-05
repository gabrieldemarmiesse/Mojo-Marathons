from algorithm.functional import vectorize, parallelize


fn matmul[
    Type: DType, M: Int, N: Int, K: Int, //
](inout C: Matrix[Type, M, N], A: Matrix[Type, M, K], B: Matrix[Type, K, N]):
    alias nelts = simdwidthof[Type]() * 2
    print(Type, nelts, nelts * 4)

    @parameter
    fn calc_row(m: Int):
        alias tile_size = 4
        alias tile_x = nelts * tile_size
        alias tile_y = tile_size
        for y in range(0, N, tile_y):
            for x in range(0, K, tile_x):
                for k in range(y, y + tile_y):

                    @parameter
                    fn dot[nelts: Int](n: Int):
                        C.store(
                            m,
                            n + x,
                            C.load[nelts](m, n + x)
                            + A[m, k] * B.load[nelts](k, n + x),
                        )

                    vectorize[dot, nelts, size=tile_x]()

    parallelize[calc_row](M, 8)
    print("done")
