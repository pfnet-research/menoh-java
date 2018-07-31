package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.jupiter.api.Test;

public class ModelBuilderTest {
    @Test
    public void makeModelBuilderIsSuccessful() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = ModelBuilder.make(vpt)
        ) {
            modelBuilder.attach("input", inputData);

            assertAll("model builder",
                    () -> assertNotNull(modelBuilder.nativeHandle()),
                    () -> assertNotNull(modelBuilder.attachedBuffers()),
                    () -> assertTrue(!modelBuilder.attachedBuffers().isEmpty(),
                            "attachedBuffers should not be empty")
            );
        }
    }

    @Test
    public void closeModelBuilder() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
        ) {
            final ModelBuilder modelBuilder = ModelBuilder.make(vpt);
            try {
                modelBuilder.attach("input", inputData);

                assertAll("model builder",
                        () -> assertNotNull(modelBuilder.nativeHandle()),
                        () -> assertNotNull(modelBuilder.attachedBuffers()),
                        () -> assertTrue(!modelBuilder.attachedBuffers().isEmpty(),
                                "attachedBuffers should not be empty")
                );
            } finally {
                modelBuilder.close();
                assertAll("model builder",
                        () -> assertNull(modelBuilder.nativeHandle()),
                        () -> assertNotNull(modelBuilder.attachedBuffers()),
                        () -> assertTrue(modelBuilder.attachedBuffers().isEmpty(),
                                "attachedBuffers should be empty")
                );

                // close() is an idempotent operation
                modelBuilder.close();
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
            assertTrue(!modelBuilder.attachedBuffers().isEmpty(), "attachedBuffers should not be empty");

            try (Model model = modelBuilder.build(modelData, backendName, backendConfig)) {
                assertAll("model",
                        () -> assertNotNull(model.nativeHandle()),
                        () -> assertNotNull(model.attachedBuffers()),
                        () -> assertArrayEquals(
                                modelBuilder.attachedBuffers().toArray(), model.attachedBuffers().toArray())
                );
            }
        }
    }

    @Test
    public void closeModelBeforeModelBuilder() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] input = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final String backendName = "mkldnn";
        final String backendConfig = "";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = ModelBuilder.make(vpt)
        ) {
            modelBuilder.attach("input", input);
            assertTrue(!modelBuilder.attachedBuffers().isEmpty(), "attachedBuffers should not be empty");

            final Model model = modelBuilder.build(modelData, backendName, backendConfig);
            try {
                assertAll("model",
                        () -> assertNotNull(model.nativeHandle()),
                        () -> assertNotNull(model.attachedBuffers()),
                        () -> assertArrayEquals(
                                modelBuilder.attachedBuffers().toArray(), model.attachedBuffers().toArray())
                );
            } finally {
                model.close();
                assertAll("model",
                        () -> assertNull(model.nativeHandle()),
                        () -> assertNotNull(model.attachedBuffers()),
                        () -> assertTrue(model.attachedBuffers().isEmpty(), "attachedBuffers should be empty")
                );
                assertAll("model builder",
                        () -> assertNotNull(modelBuilder.attachedBuffers()),
                        () -> assertTrue(!modelBuilder.attachedBuffers().isEmpty(),
                                "attachedBuffers should not be empty")
                );

                // close() is an idempotent operation
                model.close();
            }
        }
    }

    @Test
    public void closeModelBuilderBeforeModel() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] input = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final String backendName = "mkldnn";
        final String backendConfig = "";

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            try (ModelBuilder modelBuilder = ModelBuilder.make(vpt)) {
                modelBuilder.attach("input", input);
                assertTrue(!modelBuilder.attachedBuffers().isEmpty(), "attachedBuffers should not be empty");

                final Model model = modelBuilder.build(modelData, backendName, backendConfig);
                try {
                    // close model builder before model
                    modelBuilder.close();

                    assertAll("model builder",
                            () -> assertNotNull(modelBuilder.attachedBuffers()),
                            () -> assertTrue(modelBuilder.attachedBuffers().isEmpty(),
                                    "attachedBuffers should be empty")
                    );
                    assertAll("model",
                            () -> assertNotNull(model.nativeHandle()),
                            () -> assertNotNull(model.attachedBuffers()),
                            () -> assertTrue(!model.attachedBuffers().isEmpty(),
                                    "attachedBuffers should not be empty")
                    );
                } finally {
                    model.close();
                }
            }
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
    public void buildAndRunModelIfInputIsDirectBuffer() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final ByteBuffer inputDataBuf =
                ByteBuffer.allocateDirect(inputData.length * 4).order(ByteOrder.nativeOrder()); // test case
        inputDataBuf.asFloatBuffer().put(inputData); // should be native order
        final int outputDim = 1;
        final float[] expectedOutput = new float[] {0f, 0f, 0f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = makeModelBuilderWithInput(vpt, "input", inputDataBuf);
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
    public void buildAndRunModelIfInputIsArrayBackedBuffer() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final ByteBuffer inputDataBuf =
                ByteBuffer.allocate(inputData.length * 4).order(ByteOrder.nativeOrder()); // test case
        inputDataBuf.asFloatBuffer().put(inputData);
        final int outputDim = 1;
        final float[] expectedOutput = new float[] {0f, 0f, 0f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = makeModelBuilderWithInput(vpt, "input", inputDataBuf);
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
    public void buildAndRunModelIfInputIsReadOnlyBuffer() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final ByteBuffer inputDataBuf =
                ByteBuffer.allocate(inputData.length * 4).order(ByteOrder.nativeOrder());
        inputDataBuf.asFloatBuffer().put(inputData);
        final ByteBuffer readOnlyInputDataBuf = inputDataBuf.asReadOnlyBuffer(); // test case
        final int outputDim = 1;
        final float[] expectedOutput = new float[] {0f, 0f, 0f, 1f};

        try (
                ModelData modelData = ModelData.makeFromOnnx(path);
                VariableProfileTableBuilder vptBuilder = makeVptBuilderForAndModel(new int[] {batchSize, inputDim});
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = makeModelBuilderWithInput(vpt, "input", readOnlyInputDataBuf);
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
    public void buildAndRunModelIfInputIsFloatArray() throws Exception {
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
}
