package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class VariableProfileTableBuilderTest {
    @Test
    public void makeVptBuilderIsSuccessful() {
        final VariableProfileTableBuilder builder = VariableProfileTableBuilder.make();
        try {
            assertNotNull(builder.nativeHandle());
        } finally {
            builder.close();
            assertNull(builder.nativeHandle());

            // close() is an idempotent operation
            builder.close();
        }
    }

    @Test
    public void addValidInputProfile() {
        try (VariableProfileTableBuilder builder = VariableProfileTableBuilder.make()) {
            builder.addInputProfile("foo", DType.FLOAT, new int[] {1, 1});
        }
    }

    @Test
    public void addValidInputProfileWithInvalidDims() {
        try (VariableProfileTableBuilder builder = VariableProfileTableBuilder.make()) {
            MenohException e = assertThrows(MenohException.class,
                    // test case: dims.length == 1
                    () -> builder.addInputProfile("foo", DType.FLOAT, new int[] {1, 1, 1, 1, 1}));
            assertAll("invalid dim size",
                    () -> assertEquals(ErrorCode.UNDEFINED, e.getErrorCode()),
                    () -> assertTrue(e.getMessage().startsWith("foo has an invalid dims size: 5"),
                            String.format("%s doesn't start with \"foo has an invalid dims size: 5\"",
                                    e.getMessage())));
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
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final int outputDim = 1;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            assertNotNull(vpt.nativeHandle());

            // check variable profiles
            VariableProfile inputVp = vpt.variableProfile("input");
            assertAll("input variable profile",
                    () -> assertEquals(DType.FLOAT, inputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVp.dims())
            );
            VariableProfile outputVp = vpt.variableProfile("output");
            assertAll("output variable profile",
                    () -> assertEquals(DType.FLOAT, outputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVp.dims())
            );
        }
    }

    @Test
    public void buildVariableProfileTableWithBatchedInput() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 16; // test case
        final int inputDim = 2;
        final int outputDim = 1;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            assertNotNull(vpt.nativeHandle());

            // check variable profiles
            VariableProfile inputVp = vpt.variableProfile("input");
            assertAll("input variable profile",
                    () -> assertEquals(DType.FLOAT, inputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVp.dims())
            );
            VariableProfile outputVp = vpt.variableProfile("output");
            assertAll("output variable profile",
                    () -> assertEquals(DType.FLOAT, outputVp.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVp.dims())
            );
        }
    }

    @Test
    public void closeVariableProfileTable() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim})
        ) {
            final VariableProfileTable vpt = vptBuilder.build(modelData);
            try {
                assertNotNull(vpt.nativeHandle());
            } finally {
                vpt.close();
                assertNull(vpt.nativeHandle());

                // close() is an idempotent operation
                vpt.close();
            }
        }
    }

    @Test
    public void buildVariableProfileTableIfInputProfileNameNotFound() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final String inputProfileName = "__non_existent_variable__"; // test case
        final String inputVariableNameInModel = "input";
        final String outputProfileName = "output";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        makeVptBuilderForAndModel(
                            inputProfileName, new int[] {batchSize, inputDim}, outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("input profile name not found",
                    () -> assertEquals(ErrorCode.VARIABLE_NOT_FOUND, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh variable not found error: %s (variable_not_found)",
                                    inputVariableNameInModel), // not `inputProfileName`
                            e.getMessage())
            );
        }
    }

    @Test
    public void buildVariableProfileTableIfInputProfileDimsMismatched() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 3; // test case
        final String inputProfileName = "input";
        final String outputProfileName = "output";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        makeVptBuilderForAndModel(
                                inputProfileName, new int[] {batchSize, inputDim}, outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("mismatched input dims",
                    () -> assertEquals(ErrorCode.DIMENSION_MISMATCH, e.getErrorCode()),
                    () -> assertEquals(
                            String.format(
                                    "menoh dimension mismatch error: Gemm issuing \"%s\": input[1] and weight[1] "
                                            + "actual value: %d valid value: %d (dimension_mismatch)",
                                    "140211424823896", // the name of output[0] in Gemm layer
                                    3, 2),
                            e.getMessage())
            );
        }
    }

    @Test
    public void buildVariableProfileTableIfOutputProfileNameNotFound() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final String inputProfileName = "input";
        final String outputProfileName = "__non_existent_variable__"; // test case

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder =
                        makeVptBuilderForAndModel(
                            inputProfileName, new int[] {batchSize, inputDim}, outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("output profile name not found",
                    () -> assertEquals(ErrorCode.VARIABLE_NOT_FOUND, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh variable not found error: %s (variable_not_found)", outputProfileName),
                            e.getMessage())
            );
        }
    }
}
