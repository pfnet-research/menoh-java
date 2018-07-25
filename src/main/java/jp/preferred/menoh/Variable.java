package jp.preferred.menoh;

import com.sun.jna.Pointer;

import java.nio.ByteBuffer;

public class Variable {
    private final DType dtype;
    private final int[] dims;
    private final Pointer bufferHandle;

    Variable(DType dtype, int[] dims, Pointer bufferHandle) {
        this.dtype = dtype;
        this.dims = dims;
        this.bufferHandle = bufferHandle;
    }

    public DType dtype() {
        return this.dtype;
    }

    public int[] dims() {
        if (dims != null) {
            return this.dims.clone();
        } else {
            return new int[0];
        }
    }

    public ByteBuffer buffer() throws MenohException {
        final long offset = 0;
        final long elementSize = dtype.size();

        long length;
        if (dims.length > 0) {
            length = 1;
            for (int d : dims) {
                length *= d;
            }
        } else {
            length = 0;
        }

        return this.bufferHandle.getByteBuffer(offset, elementSize * length);
    }
}
