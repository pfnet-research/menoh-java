package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class ModelBuilderTest {
    @Test
    public void makeModelBuilderIsSuccessful() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            ModelBuilder modelBuilder = null;
            try {
                modelBuilder = ModelBuilder.make(vpt);

                assertNotNull(modelBuilder);
                assertNotNull(modelBuilder.nativeHandle());
            } finally {
                if (modelBuilder != null) {
                    modelBuilder.close();
                    assertNull(modelBuilder.nativeHandle());

                    // close() is an idempotent operation
                    modelBuilder.close();
                }
            }
        }
    }

    @Test
    public void closeModelBuilderIsIdempotent() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
        ) {
            ModelBuilder modelBuilder = null;
            try {
                modelBuilder = ModelBuilder.make(vpt);

                assertNotNull(modelBuilder);
                assertNotNull(modelBuilder.nativeHandle());

                modelBuilder.attach("input", inputData);
            } finally {
                if (modelBuilder != null) {
                    modelBuilder.close();
                    assertNull(modelBuilder.nativeHandle());

                    // close() is an idempotent operation
                    modelBuilder.close();
                }
            }
        }
    }

    @Test
    public void buildModelIsSuccessful() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final String backendName = "mkldnn";
        final String backendConfig = "";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = ModelBuilder.make(vpt)
        ) {
            modelBuilder.attach("input", new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f});
            modelBuilder.build(modelData, backendName, backendConfig);
        }
    }

    @Test
    public void buildModelIfBackendNotFound() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final String backendName = "__non_existent_backend__"; // test case
        final String backendConfig = "";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = ModelBuilder.make(vpt)
        ) {
            modelBuilder.attach("input", new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f});

            MenohException e = assertThrows(
                    MenohException.class, () -> modelBuilder.build(modelData, backendName, backendConfig));
            assertAll("backendName is invalid",
                    () -> assertEquals(ErrorCode.INVALID_BACKEND_NAME, e.getErrorCode()),
                    () -> assertEquals(
                            String.format("menoh invalid backend name error: %s (invalid_backend_name)", backendName),
                            e.getMessage())
            );
        }
    }

    @Test
    public void buildAndRunModelIsSuccessful() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final int outputDim = 1;
        final float[] expectedOutput = new float[] {0f, 0f, 0f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = makeModelBuilderWithInput(vpt, "input", inputData);
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            model.run();

            // you can delete modelData explicitly after model building
            modelData.close();

            final Variable inputVar = model.variable("input");
            assertAll("input variable",
                    () -> assertEquals(DType.FLOAT, inputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVar.dims())
            );
            final int[] inputDims = inputVar.dims();
            final float[] inputBuf = new float[inputDims[0] * inputDims[1]];
            inputVar.buffer().asFloatBuffer().get(inputBuf);
            assertArrayEquals(inputData, inputBuf);

            final Variable outputVar = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar.dims())
            );
            final float[] outputBuf = new float[outputVar.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf);
            assertArrayEquals(expectedOutput, outputBuf);
        }
    }

    @Test
    public void closeModelIsIdempotent() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] input = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = ModelBuilder.make(vpt)
        ) {
            modelBuilder.attach("input", input);

            Model model = null;
            try {
                model = modelBuilder.build(modelData, "mkldnn", "");

                assertNotNull(model);
                assertNotNull(model.nativeHandle());

                model.run();
            } finally {
                if (model != null) {
                    model.close();
                    assertNull(model.nativeHandle());

                    // close() is an idempotent operation
                    model.close();
                }
            }
        }
    }
}
