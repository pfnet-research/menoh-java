package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class ModelDataTest {
    @Test
    public void makeFromValidOnnxFile() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");

        try (ModelData modelData = ModelData.makeFromOnnx(path)) {
            assertNotNull(modelData.nativeHandle());
        }
    }

    @Test
    public void closeModelDataIsIdempotent() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");

        ModelData modelData = null;
        try {
            modelData = ModelData.makeFromOnnx(path);
        } finally {
            if (modelData != null) {
                modelData.close();
                assertNull(modelData.nativeHandle());

                // close() is an idempotent operation
                modelData.close();
            }
        }
    }

    @Test
    public void makeFromNonExistentOnnxFile() {
        MenohException e = assertThrows(
                MenohException.class, () -> ModelData.makeFromOnnx("__NON_EXISTENT_FILENAME__"));
        assertAll("non-existent onnx file",
                () -> assertEquals(ErrorCode.INVALID_FILENAME, e.getErrorCode()),
                () -> assertEquals(
                        "menoh invalid filename error: __NON_EXISTENT_FILENAME__ (invalid_filename)",
                        e.getMessage())
        );
    }

    @Test
    public void makeFromInvalidOnnxFile() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/invalid_format.onnx");

        MenohException e = assertThrows(MenohException.class, () -> ModelData.makeFromOnnx(path));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.ONNX_PARSE_ERROR, e.getErrorCode()),
                () -> assertEquals(
                        String.format("menoh onnx parse error: %s (onnx_parse_error)", path),
                        e.getMessage())
        );
    }

    @Test
    public void makeFromUnsupportedOnnxOpsetVersionFile() throws Exception {
        // Note: This file is a copy of and.onnx which is edited the last byte to 127 (0x7e)
        final String path = TestUtils.getResourceFilePath("models/unsupported_onnx_opset_version.onnx");

        MenohException e = assertThrows(MenohException.class, () -> ModelData.makeFromOnnx(path));
        assertAll("invalid onnx file",
                () -> assertEquals(ErrorCode.UNSUPPORTED_ONNX_OPSET_VERSION, e.getErrorCode()),
                () -> assertEquals(
                        String.format(
                                "menoh unsupported onnx opset version error: %s has "
                                        + "onnx opset version %d > %d (unsupported_onnx_opset_version)",
                                path, 127, 7),
                        e.getMessage())
        );
    }

    @Test
    public void optimizeModelData() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = TestUtils.makeVptBuilderForAndModel(new int[] {1, 2});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            modelData.optimize(vpt);
        }
    }
}