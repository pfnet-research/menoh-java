package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class ModelDataTest {
    @Test
    public void makeFromValidOnnxFile() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");

        try (ModelData modelData = ModelData.fromOnnxFile(path)) {
            assertNotNull(modelData.nativeHandle());
        }
    }

    @Test
    public void closeModelData() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");

        final ModelData modelData = ModelData.fromOnnxFile(path);
        try {
            assertNotNull(modelData.nativeHandle());
        } finally {
            modelData.close();
            assertNull(modelData.nativeHandle());

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
