package jp.preferred.menoh;

import com.sun.jna.Pointer;

public class Variable {
    private final DType dtype;
    private final int[] dims;
    private final Pointer bufferHandle;

    Variable(DType dtype, int[] dims, Pointer bufferHandle) {
        this.dtype = dtype;
        this.dims = dims;
        this.bufferHandle = bufferHandle;
    }

    DType dtype() {
        return this.dtype;
    }

    int[] dims() {
        return this.dims;
    }

    Pointer bufferHandle() {
        return this.bufferHandle;
    }
}
