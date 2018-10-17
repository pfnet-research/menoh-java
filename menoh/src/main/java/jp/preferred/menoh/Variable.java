package jp.preferred.menoh;

import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * An input or output variable in the model.
 */
public class Variable {
    private final DType dtype;
    private final int[] dims;
    private final Pointer bufferHandle;

    Variable(DType dtype, int[] dims, Pointer bufferHandle) {
        this.dtype = dtype;
        this.dims = dims;
        this.bufferHandle = bufferHandle;
    }

    /**
     * A data type of the variable.
     */
    public DType dtype() {
        return this.dtype;
    }

    /**
     * An array of dimensions.
     */
    public int[] dims() {
        if (dims != null) {
            return this.dims.clone();
        } else {
            return new int[0];
        }
    }

    /**
     * <p>A direct {@link ByteBuffer} which points to the native buffer of the variable. The buffer can be read
     * and written via the methods of <code>ByteBuffer</code> before and after running the model.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer is {@link ByteOrder#nativeOrder()} which may differ
     * from JVM.</p>
     *
     * <p>e.g.:</p>
     * <pre>
     * {@code
     * final ModelRunner runner = ...;
     * final ByteBuffer data = ...;
     *
     * final Variable input1 = runner.variable("input1");
     * final ByteBuffer buf = input1.buffer();
     * buf.clear();
     * buf.put(data.duplicate()).rewind();
     *
     * runner.run();
     *
     * final Variable output1 = runner.variable("output1");
     * final ByteBuffer ret = output1.buffer();
     * ...
     * }
     * </pre>
     */
    public ByteBuffer buffer() throws MenohException {
        return this.bufferHandle.getByteBuffer(0, bufferLength());
    }

    /**
     * The length of the buffer in bytes.
     */
    long bufferLength() {
        if (dims.length > 0) {
            final long elementSize = dtype.size();
            long length = 1;
            for (int d : dims) {
                length *= d;
            }

            if (length <= 0) {
                throw new MenohException(ErrorCode.UNDEFINED, "buffer is empty");
            }

            return elementSize * length;
        } else {
            throw new MenohException(ErrorCode.UNDEFINED, "buffer is empty");
        }
    }
}
