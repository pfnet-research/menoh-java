package jp.preferred.menoh;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class BufferUtils {
    /**
     * <p>Copies a buffer to a native memory.</p>
     *
     * <p>If the <code>buffer</code> is direct, it will be attached to the model directly without copying.
     * Otherwise, it copies the content of the buffer to a newly allocated memory in the native heap ranging
     * from <code>position()</code> to <code>(limit() - 1)</code> without changing its position.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffer the non-empty byte buffer from which to copy
     *
     * @return the pointer to the allocated native memory
     * @throws IllegalArgumentException if <code>buffer</code> is null or empty
     */
    static Pointer copyToNativeMemory(final ByteBuffer buffer) {
        if (buffer == null || buffer.remaining() <= 0) {
            throw new IllegalArgumentException("buffer must not be null or empty");
        }

        final int length = buffer.remaining();

        if (buffer.isDirect()) {
            final int offset = buffer.position();

            // return a pointer to the direct buffer without copying
            return Native.getDirectBufferPointer(buffer).share(offset, length);
        } else {
            final Memory mem = new Memory(length);

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

            return mem.share(0, length);
        }
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
    static Pointer copyToNativeMemory(final float[] values, final int offset, final int length) {
        if (values == null || values.length <= 0) {
            throw new IllegalArgumentException("values must not be null or empty");
        }

        final Memory mem = new Memory((long) length * 4);
        mem.write(0, values, offset, length);

        return mem.share(0, length);
    }
}
