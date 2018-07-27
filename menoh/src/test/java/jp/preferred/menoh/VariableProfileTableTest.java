package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class VariableProfileTableTest {
    @Test
    public void makeVptBuilder() {
        VariableProfileTableBuilder builder = null;
        try {
            builder = VariableProfileTableBuilder.make();

            assertNotNull(builder);
            assertNotNull(builder.nativeHandle());
        } finally {
            if (builder != null) {
                builder.close();
                assertNull(builder.nativeHandle());

                // close() is an idempotent operation
                builder.close();
            }
        }
    }

    @Test
    public void addValidInputProfile() {
        try (VariableProfileTableBuilder builder = VariableProfileTableBuilder.make()) {
            builder.addInputProfile("foo", DType.FLOAT, new int[] {1, 1});
        }
    }

    @Test
    public void addValidOutputProfile() {
        try (VariableProfileTableBuilder builder = VariableProfileTableBuilder.make()) {
            builder.addOutputProfile("foo", DType.FLOAT);
        }
    }

    @Test
    public void buildVariableProfileTable() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final int outputDim = 1;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        TestUtils.makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            assertNotNull(vpt.nativeHandle());

            // check variables
            VariableProfile inputVp = vpt.variableProfile("input");
            assertAll("input variable profile",
                    () -> assertEquals(DType.FLOAT, inputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVp.dims())
            );
            VariableProfile fc1Vp = vpt.variableProfile("fc1");
            assertAll("fc1 variable profile",
                    () -> assertEquals(DType.FLOAT, fc1Vp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, fc1Vp.dims())
            );
        }
    }

    @Test
    public void buildVariableProfileTableWithBatchedInput() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");
        final int batchSize = 16; // test case
        final int inputDim = 2;
        final int outputDim = 1;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        TestUtils.makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            assertNotNull(vpt.nativeHandle());

            // check variables
            VariableProfile inputVp = vpt.variableProfile("input");
            assertAll("input variable profile",
                    () -> assertEquals(DType.FLOAT, inputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVp.dims())
            );
            VariableProfile fc1Vp = vpt.variableProfile("fc1");
            assertAll("fc1 variable profile",
                    () -> assertEquals(DType.FLOAT, fc1Vp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, fc1Vp.dims())
            );
        }
    }

    @Test
    public void closeVariableProfileTableIsIdempotent() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");
        final int batchSize = 1;
        final int inputDim = 2;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        TestUtils.makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            assertNotNull(vpt.nativeHandle());
        }
    }

    @Test
    public void buildVariableProfileTableWithMismatchedInputProfileDims() throws Exception {
        final String path = TestUtils.getResourceFilePath("models/and.onnx");
        final int batchSize = 1;
        final int inputDim = 3; // test case

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        TestUtils.makeVptBuilderForAndModel(new int[] {batchSize, inputDim})
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("mismatched input dims",
                    () -> assertEquals(ErrorCode.DIMENSION_MISMATCH, e.getErrorCode()),
                    () -> assertEquals(
                            String.format(
                                    "menoh dimension mismatch error: Gemm issuing \"%s\": input[1] and weight[1] "
                                            + "actual value: %d valid value: %d (dimension_mismatch)",
                                    "140388529908816", // the name of output[0]
                                    3, 2),
                            e.getMessage())
            );
        }
    }
}
