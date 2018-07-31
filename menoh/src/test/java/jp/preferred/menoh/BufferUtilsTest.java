package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import com.sun.jna.Pointer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.jupiter.api.Test;

public class BufferUtilsTest {
    @Test
    public void copyDirectBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};

        final ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length);
        buf.put(bytes).rewind();
        assertEquals(8, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(bytes, ptr.getByteArray(0, bytes.length));
        assertAll("buffer's state",
                () -> assertEquals(0, buf.position()),
                () -> assertEquals(8, buf.limit())
        );
    }

    @Test
    public void copySlicedDirectBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};
        final byte[] slicedBytes = new byte[] {2, 3, 4, 5};

        final ByteBuffer buf = ByteBuffer.allocateDirect(bytes.length);
        buf.put(bytes).rewind();

        // slice 4 bytes
        buf.position(2).limit(6);
        assertEquals(4, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(slicedBytes, ptr.getByteArray(0, slicedBytes.length));
        assertAll("buffer's state",
                () -> assertEquals(2, buf.position()),
                () -> assertEquals(6, buf.limit())
        );
    }

    @Test
    public void copyEmptyDirectBufferToNativeMemory() {
        final ByteBuffer buf = ByteBuffer.allocateDirect(0);
        assertEquals(0, buf.remaining());

        assertThrows(
                IllegalArgumentException.class,
                () -> BufferUtils.copyToNativeMemory(buf));
    }

    @Test
    public void copyArrayBackedBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};

        final ByteBuffer buf = ByteBuffer.wrap(bytes);
        assertEquals(8, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(bytes, ptr.getByteArray(0, bytes.length));
        assertAll("buffer's state",
                () -> assertEquals(0, buf.position()),
                () -> assertEquals(8, buf.limit())
        );
    }

    @Test
    public void copySlicedArrayBackedBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};
        final byte[] slicedBytes = new byte[] {2, 3, 4, 5};

        final ByteBuffer buf = ByteBuffer.wrap(bytes, 1, 6).slice();
        assertAll("non-zero array offset",
                () -> assertEquals(1, buf.arrayOffset()),
                () -> assertEquals(0, buf.position()),
                () -> assertEquals(6, buf.limit())
        );

        // slice 4 bytes
        buf.position(1).limit(5);
        assertEquals(4, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(slicedBytes, ptr.getByteArray(0, slicedBytes.length));
        assertAll("buffer's state",
                () -> assertEquals(1, buf.position()),
                () -> assertEquals(5, buf.limit())
        );
    }

    @Test
    public void copyReadOnlyBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};

        final ByteBuffer buf = ByteBuffer.wrap(bytes).asReadOnlyBuffer();
        assertEquals(8, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(bytes, ptr.getByteArray(0, bytes.length));
        assertAll("buffer's state",
                () -> assertEquals(0, buf.position()),
                () -> assertEquals(8, buf.limit())
        );
    }

    @Test
    public void copySlicedReadOnlyBufferToNativeMemory() {
        final byte[] bytes = new byte[] {0, 1, 2, 3, 4, 5, 6, 7};
        final byte[] slicedBytes = new byte[] {2, 3, 4, 5};

        final ByteBuffer buf = ByteBuffer.wrap(bytes).asReadOnlyBuffer();

        // slice 4 bytes
        buf.position(2).limit(6);
        assertEquals(4, buf.remaining());

        final Pointer ptr = BufferUtils.copyToNativeMemory(buf);
        assertNotNull(ptr);
        assertArrayEquals(slicedBytes, ptr.getByteArray(0, slicedBytes.length));
        assertAll("buffer's state",
                () -> assertEquals(2, buf.position()),
                () -> assertEquals(6, buf.limit())
        );
    }

    @Test
    public void copyFloatArrayToNativeMemory() {
        final float[] values = new float[] {0f, 1f, 2f, 3f};
        final ByteBuffer valuesBuf = ByteBuffer.allocate(values.length * 4).order(ByteOrder.nativeOrder());
        valuesBuf.asFloatBuffer().put(values);
        final int offset = 0;
        final int length = values.length;

        final Pointer ptr = BufferUtils.copyToNativeMemory(values, offset, length);
        assertNotNull(ptr);
        assertAll("copied values in native memory",
                () -> assertArrayEquals(values, ptr.getFloatArray(0, length)),
                // check the byte order
                () -> assertArrayEquals(valuesBuf.array(), ptr.getByteArray(0, length * 4))
        );
    }

    @Test
    public void copySlicedFloatArrayToNativeMemory() {
        final float[] values = new float[] {0f, 1f, 2f, 3f};
        final float[] slicedValues = new float[] {1f, 2f};
        final ByteBuffer slicedValuesBuf = ByteBuffer.allocate(slicedValues.length * 4).order(ByteOrder.nativeOrder());
        slicedValuesBuf.asFloatBuffer().put(slicedValues);
        final int offset = 1;
        final int length = slicedValues.length;

        // slice 2 elements (8 bytes)
        final Pointer ptr = BufferUtils.copyToNativeMemory(values, offset, length);
        assertNotNull(ptr);
        assertAll("copied values in native memory",
                () -> assertArrayEquals(slicedValues, ptr.getFloatArray(0, length)),
                // check the byte order
                () -> assertArrayEquals(slicedValuesBuf.array(), ptr.getByteArray(0, length * 4))
        );
    }

    @Test
    public void copyEmptyFloatArrayToNativeMemory() {
        final float[] values = new float[] {}; // test case
        final int offset = 0;
        final int length = values.length;

        assertThrows(
                IllegalArgumentException.class,
                () -> BufferUtils.copyToNativeMemory(values, offset, length));
    }

    @Test
    public void copyNullFloatArrayToNativeMemory() {
        final float[] values = null; // test case
        final int offset = 0;
        final int length = 0;

        assertThrows(
                IllegalArgumentException.class,
                () -> BufferUtils.copyToNativeMemory(values, offset, length));
    }

    @Test
    public void copyNegativeOffsetFloatArrayToNativeMemory() {
        final float[] values = new float[] {0f, 1f, 2f, 3f};
        final int offset = -1; // test case
        final int length = values.length;

        assertThrows(
                ArrayIndexOutOfBoundsException.class,
                () -> BufferUtils.copyToNativeMemory(values, offset, length));
    }

    @Test
    public void copyNegativeLengthFloatArrayToNativeMemory() {
        final float[] values = new float[] {0f, 1f, 2f, 3f};
        final int offset = 0;
        final int length = -1; // test case

        assertThrows(
                IllegalArgumentException.class,
                () -> BufferUtils.copyToNativeMemory(values, offset, length));
    }
}
