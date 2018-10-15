package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class VariableProfileTableTest {
    @Test
    public void makeVptBuilderIsSuccessful() {
        final VariableProfileTableBuilder builder = VariableProfileTable.builder();
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
    public void addValidInputProfileDims2() {
        try (VariableProfileTableBuilder builder = VariableProfileTable.builder()) {
            builder.addInputProfile("foo", DType.FLOAT, new int[] {1, 1});
        }
    }

    @Test
    public void addValidInputProfileDims5() {
        try (VariableProfileTableBuilder builder = VariableProfileTable.builder()) {
            builder.addInputProfile("foo", DType.FLOAT, new int[] {1, 1, 1, 1, 1});
        }
    }

    @Test
    public void addValidOutputProfile() {
        try (VariableProfileTableBuilder builder = VariableProfileTable.builder()) {
            builder.addOutputName("foo");
        }
    }

    @Test
    public void buildVariableProfileTable() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final int outputDim = 1;

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName("output");
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName("output");
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName("output")
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
    public void buildVariableProfileTableIfVariableNotFound() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final String inputVariableNameInModel = "input";
        final String outputProfileName = "output";

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        // test case (no addInputProfile for "input" variable)
                        .addOutputName(outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("input profile name not found",
                    () -> assertEquals(ErrorCode.VARIABLE_NOT_FOUND, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh variable not found error: %s (variable_not_found)",
                                    inputVariableNameInModel),
                            e.getMessage())
            );
        }
    }

    @Test
    public void buildVariableProfileTableIfInputProfileNameNotFound() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final String inputProfileName = "__non_existent_variable__"; // test case
        final String outputProfileName = "output";

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile(inputProfileName, DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName(outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("input profile name not found",
                    () -> assertEquals(ErrorCode.INPUT_NOT_FOUND_ERROR, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh input not found error: %s (input_not_found_error)",
                                    inputProfileName),
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile(inputProfileName, DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName(outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("mismatched input dims",
                    () -> assertEquals(ErrorCode.DIMENSION_MISMATCH, e.getErrorCode()),
                    () -> assertEquals(
                            String.format(
                                    "menoh dimension mismatch error: Gemm issuing \"input\": trans(A)[1] and "
                                            + "trans(B)[0]) actual value: %d valid value: %d (dimension_mismatch)",
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile(inputProfileName, DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName(outputProfileName)
        ) {
            MenohException e = assertThrows(MenohException.class, () -> vptBuilder.build(modelData));
            assertAll("output profile name not found",
                    () -> assertEquals(ErrorCode.OUTPUT_NOT_FOUND_ERROR, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh output not found error: %s (output_not_found_error)",
                                    outputProfileName),
                            e.getMessage())
            );
        }
    }
}
