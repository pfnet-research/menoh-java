package jp.preferred.menoh;

import com.sun.jna.Memory;
import com.sun.jna.Native;
import com.sun.jna.Pointer;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

class BufferUtils {
    static Pointer copyToNativeMemory(final ByteBuffer buffer) {
        if (buffer.isDirect()) {
            // pass a pointer to the direct buffer directly instead of copying
            return Native.getDirectBufferPointer(buffer);
        } else {
            final int length = buffer.remaining();
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

    static ByteBuffer toDirectByteBuffer(float[] values, int offset, int length) {
        final ByteBuffer buf = ByteBuffer.allocateDirect(length * 4).order(ByteOrder.nativeOrder());
        buf.asFloatBuffer().put(values, offset, length);

        return buf;
    }
}
