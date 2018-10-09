package jp.preferred.menoh;

import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class BufferUtils {
    /**
     * <p>If the <code>buffer</code> is direct, it will be converted to the pointer directly without copying.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffer the non-empty direct byte buffer
     *
     * @return the pointer to the native memory
     * @throws IllegalArgumentException if <code>buffer</code> is not direct
     */
    static Pointer convertToNativeMemory(final ByteBuffer buffer) {
        if (buffer.isDirect()) {
            final int offset = buffer.position();
            final int length = buffer.remaining();

            // return a pointer to the direct buffer without copying
            return Native.getDirectBufferPointer(buffer).share(offset, length);
        } else {
            throw new IllegalArgumentException("buffer must be direct");
        }
    }

    /**
     * <p>Copies a buffer to a native memory.</p>
     *
     * <p>It copies the content of the buffer to a newly allocated memory in the native heap ranging from
     * <code>position()</code> to <code>(limit() - 1)</code> without changing its position.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffer the non-empty byte buffer from which to copy
     *
     * @return the pointer to the allocated native memory
     * @throws IllegalArgumentException if <code>buffer</code> is null or empty
     */
    static DisposableMemory copyToNativeMemory(final ByteBuffer buffer) {
        if (buffer == null || buffer.remaining() <= 0) {
            throw new IllegalArgumentException("buffer must not be null or empty");
        }

        final int length = buffer.remaining();
        final DisposableMemory mem = new DisposableMemory(length);

        int index;
        byte[] bytes;
        if (buffer.hasArray()) { // it is array-backed and not read-only
            index = buffer.arrayOffset() + buffer.position();
            bytes = buffer.array();
        } else {
            index = 0;
            // use duplicated buffer to avoid changing `position`
            bytes = new byte[length];
            buffer.duplicate().get(bytes);
        }
        mem.write(0, bytes, index, length);

        return mem;
    }

    /**
     * <p>Copies the array to a newly allocated memory in the native heap.</p>
     *
     * @param values the non-empty array from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     *
     * @return the pointer to the allocated native memory
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    static DisposableMemory copyToNativeMemory(final byte[] values, final int offset, final int length) {
        if (values == null || values.length <= 0) {
            throw new IllegalArgumentException("values must not be null or empty");
        }

        final DisposableMemory mem = new DisposableMemory((long) length * Native.getNativeSize(Byte.TYPE));
        mem.write(0, values, offset, length);

        return mem;
    }

    /**
     * <p>Copies the array to a newly allocated memory in the native heap.</p>
     *
     * @param values the non-empty array from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     *
     * @return the pointer to the allocated native memory
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    static DisposableMemory copyToNativeMemory(final int[] values, final int offset, final int length) {
        if (values == null || values.length <= 0) {
            throw new IllegalArgumentException("values must not be null or empty");
        }

        final DisposableMemory mem = new DisposableMemory((long) length * Native.getNativeSize(Integer.TYPE));
        mem.write(0, values, offset, length);

        return mem;
    }

    /**
     * <p>Copies the array to a newly allocated memory in the native heap.</p>
     *
     * @param values the non-empty array from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     *
     * @return the pointer to the allocated native memory
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    static DisposableMemory copyToNativeMemory(final float[] values, final int offset, final int length) {
        if (values == null || values.length <= 0) {
            throw new IllegalArgumentException("values must not be null or empty");
        }

        final DisposableMemory mem = new DisposableMemory((long) length * Native.getNativeSize(Float.TYPE));
        mem.write(0, values, offset, length);

        return mem;
    }
}
