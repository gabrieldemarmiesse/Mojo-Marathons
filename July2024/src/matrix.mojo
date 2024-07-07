from random import rand


struct Matrix[Type: DType, rows: Int, cols: Int]:
    alias Elements = rows * cols
    var data: DTypePointer[Type]

    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[Type].alloc(Self.Elements)
        rand(data, Self.Elements)
        return Self(data)

    fn __init__(inout self):
        self.data = DTypePointer[Type].alloc(Self.Elements)
        # print("pointer value", int(self.data))
        memset_zero(self.data, Self.Elements)

    fn __init__(inout self, data: DTypePointer[Type]):
        self.data = data
        # print("pointer value", int(self.data))

    fn __del__(owned self):
        # print("freeing matrix with pointer", int(self.data))
        self.data.free()

    fn __getitem__(self, y: Int, x: Int) -> Scalar[Type]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, value: Scalar[Type]):
        self.store[1](y, x, value)

    fn load[Nelts: Int](self, y: Int, x: Int) -> SIMD[Type, Nelts]:
        return self.data.load[width=Nelts](y * cols + x)

    fn store[Nelts: Int](inout self, y: Int, x: Int, value: SIMD[Type, Nelts]):
        self.data.store[width=Nelts](y * cols + x, value)

    @always_inline
    fn get_pointer(self, y: Int, x: Int) -> DTypePointer[Type]:
        return self.data + y * cols + x
