package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import java.io.InputStream;
import java.nio.ByteBuffer;

import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.Test;

public class ModelDataTest {
    @Test
    public void makeFromValidOnnxFile() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");

        try (ModelData modelData = ModelData.fromOnnxFile(path)) { // loaded from a file
            assertNotNull(modelData.nativeHandle());
        }
    }

    @Test
    public void makeFromValidByteArray() throws Exception {
        final InputStream in = getClass().getClassLoader().getResourceAsStream("models/and_op.onnx");
        final byte[] data = IOUtils.toByteArray(in);

        try (ModelData modelData = ModelData.fromOnnx(data)) { // loaded from memory
            assertNotNull(modelData.nativeHandle());
        }
    }

    @Test
    public void makeFromValidByteBuffer() throws Exception {
        final InputStream in = getClass().getClassLoader().getResourceAsStream("models/and_op.onnx");
        final ByteBuffer data = ByteBuffer.wrap(IOUtils.toByteArray(in));

        try (ModelData modelData = ModelData.fromOnnx(data)) { // loaded from memory
            assertNotNull(modelData.nativeHandle());
        }
    }

    @Test
    public void closeModelData() throws Exception {
        final InputStream in = getClass().getClassLoader().getResourceAsStream("models/and_op.onnx");
        final byte[] data = IOUtils.toByteArray(in);

        final ModelData modelData = ModelData.fromOnnx(data); // loaded from memory
        try {
            assertNotNull(modelData.nativeHandle());
            assertNotNull(modelData.nativeDataPointer());
            assertFalse(modelData.nativeDataPointer().isDisposed());
        } finally {
            modelData.close();
            assertNull(modelData.nativeHandle());
            assertNotNull(modelData.nativeDataPointer());
            assertTrue(modelData.nativeDataPointer().isDisposed());

            // close() is an idempotent operation
            modelData.close();
        }
    }

    @Test
    public void makeFromNonExistentOnnxFile() {
        MenohException e = assertThrows(
                MenohException.class, () -> ModelData.fromOnnxFile("__NON_EXISTENT_FILENAME__"));
        assertAll("non-existent onnx file",
                () -> assertEquals(ErrorCode.INVALID_FILENAME, e.getErrorCode()),
                () -> assertEquals(
                        "menoh invalid filename error: __NON_EXISTENT_FILENAME__ (invalid_filename)",
                        e.getMessage())
        );
    }

    @Test
    public void makeFromEmptyByteArray() throws Exception {
        final byte[] data = new byte[0]; // test case

        assertThrows(IllegalArgumentException.class, () -> ModelData.fromOnnx(data));
    }

    @Test
    public void makeFromEmptyByteBuffer() throws Exception {
        final ByteBuffer data = ByteBuffer.wrap(new byte[0]); // test case

        assertThrows(IllegalArgumentException.class, () -> ModelData.fromOnnx(data));
    }

    @Test
    public void makeFromInvalidOnnxFile() throws Exception {
        final String path = getResourceFilePath("models/invalid_format.onnx");

        MenohException e = assertThrows(MenohException.class, () -> ModelData.fromOnnxFile(path));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.ONNX_PARSE_ERROR, e.getErrorCode()),
                () -> assertEquals(
                        String.format("menoh onnx parse error: %s (onnx_parse_error)", path),
                        e.getMessage())
        );
    }

    @Test
    public void makeFromInvalidByteArray() throws Exception {
        final InputStream in =
                ModelDataTest.class.getClassLoader().getResourceAsStream("models/invalid_format.onnx");
        final byte[] data = IOUtils.toByteArray(in);

        MenohException e = assertThrows(MenohException.class, () -> ModelData.fromOnnx(data));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.ONNX_PARSE_ERROR, e.getErrorCode()),
                () -> assertEquals(
                        String.format("menoh onnx parse error: parse binary onnx data on memory (onnx_parse_error)"),
                        e.getMessage())
        );
    }

    @Test
    public void makeFromInvalidByteBuffer() throws Exception {
        final InputStream in =
                ModelDataTest.class.getClassLoader().getResourceAsStream("models/invalid_format.onnx");
        final ByteBuffer data = ByteBuffer.wrap(IOUtils.toByteArray(in));

        MenohException e = assertThrows(MenohException.class, () -> ModelData.fromOnnx(data));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.ONNX_PARSE_ERROR, e.getErrorCode()),
                () -> assertEquals(
                        String.format("menoh onnx parse error: parse binary onnx data on memory (onnx_parse_error)"),
                        e.getMessage())
        );
    }

    @Test
    public void makeFromUnsupportedOnnxOpsetVersionFile() throws Exception {
        // Note: This file is a copy of and_op.onnx which is edited the last byte to 127 (0x7e)
        final String path = getResourceFilePath("models/unsupported_onnx_opset_version.onnx");

        MenohException e = assertThrows(MenohException.class, () -> ModelData.fromOnnxFile(path));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.UNSUPPORTED_ONNX_OPSET_VERSION, e.getErrorCode()),
                () -> assertEquals(
                        String.format(
                                "menoh unsupported onnx opset version error: given onnx "
                                        + "has opset version %d > %d (unsupported_onnx_opset_version)",
                                127, 8),
                        e.getMessage())
        );
    }

    @Test
    public void optimizeModelData() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {1, 2})
                        .addOutputName("output");
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            modelData.optimize(vpt);
        }
    }
}
