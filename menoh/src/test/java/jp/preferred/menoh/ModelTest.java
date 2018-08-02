package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.junit.jupiter.api.Test;

public class ModelTest {
    @Test
    public void makeModelBuilderIsSuccessful() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 1;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt)
        ) {
            modelBuilder.attachExternalBuffer("input", inputData);

            assertAll("model builder",
                    () -> assertNotNull(modelBuilder.nativeHandle()),
                    () -> assertNotNull(modelBuilder.externalBuffers()),
                    () -> assertTrue(!modelBuilder.externalBuffers().isEmpty(),
                            "externalBuffers should not be empty")
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            final ModelBuilder modelBuilder = Model.builder(vpt);
            try {
                modelBuilder.attachExternalBuffer("input", inputData);

                assertAll("model builder",
                        () -> assertNotNull(modelBuilder.nativeHandle()),
                        () -> assertNotNull(modelBuilder.externalBuffers()),
                        () -> assertTrue(!modelBuilder.externalBuffers().isEmpty(),
                                "externalBuffers should not be empty")
                );
            } finally {
                modelBuilder.close();
                assertAll("model builder",
                        () -> assertNull(modelBuilder.nativeHandle()),
                        () -> assertNotNull(modelBuilder.externalBuffers()),
                        () -> assertTrue(modelBuilder.externalBuffers().isEmpty(),
                                "externalBuffers should be empty")
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt)
        ) {
            modelBuilder.attachExternalBuffer("input", new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f});
            assertTrue(!modelBuilder.externalBuffers().isEmpty(), "externalBuffers should not be empty");

            try (Model model = modelBuilder.build(modelData, backendName, backendConfig)) {
                assertAll("model",
                        () -> assertNotNull(model.nativeHandle()),
                        () -> assertNotNull(model.externalBuffers()),
                        () -> assertArrayEquals(
                                modelBuilder.externalBuffers().toArray(), model.externalBuffers().toArray())
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt)
        ) {
            modelBuilder.attachExternalBuffer("input", input);
            assertTrue(!modelBuilder.externalBuffers().isEmpty(), "externalBuffers should not be empty");

            final Model model = modelBuilder.build(modelData, backendName, backendConfig);
            try {
                assertAll("model",
                        () -> assertNotNull(model.nativeHandle()),
                        () -> assertNotNull(model.externalBuffers()),
                        () -> assertArrayEquals(
                                modelBuilder.externalBuffers().toArray(), model.externalBuffers().toArray())
                );
            } finally {
                model.close();
                assertAll("model",
                        () -> assertNull(model.nativeHandle()),
                        () -> assertNotNull(model.externalBuffers()),
                        () -> assertTrue(model.externalBuffers().isEmpty(), "externalBuffers should be empty")
                );
                assertAll("model builder",
                        () -> assertNotNull(modelBuilder.externalBuffers()),
                        () -> assertTrue(!modelBuilder.externalBuffers().isEmpty(),
                                "externalBuffers should not be empty")
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData)
        ) {
            try (ModelBuilder modelBuilder = Model.builder(vpt)) {
                modelBuilder.attachExternalBuffer("input", input);
                assertTrue(!modelBuilder.externalBuffers().isEmpty(), "externalBuffers should not be empty");

                final Model model = modelBuilder.build(modelData, backendName, backendConfig);
                try {
                    // close model builder before model
                    modelBuilder.close();

                    assertAll("model builder",
                            () -> assertNotNull(modelBuilder.externalBuffers()),
                            () -> assertTrue(modelBuilder.externalBuffers().isEmpty(),
                                    "externalBuffers should be empty")
                    );
                    assertAll("model",
                            () -> assertNotNull(model.nativeHandle()),
                            () -> assertNotNull(model.externalBuffers()),
                            () -> assertTrue(!model.externalBuffers().isEmpty(),
                                    "externalBuffers should not be empty")
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt)
        ) {
            modelBuilder.attachExternalBuffer("input", new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f});

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
    public void buildAndRunModelWithoutAttachingInput() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData1 = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final float[] inputData2 = new float[] {1f, 1f, 1f, 0f, 0f, 1f, 0f, 0f};
        final int outputDim = 1;
        final float[] expectedOutput1 = new float[] {0f, 0f, 0f, 1f};
        final float[] expectedOutput2 = new float[] {1f, 0f, 0f, 0f};

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt); // test case
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            // you can delete modelData explicitly after building a model
            modelData.close();

            final Variable inputVar = model.variable("input");
            assertAll("input variable",
                    () -> assertEquals(DType.FLOAT, inputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVar.dims())
            );

            // write input data to the internal buffer allocated by native Menoh
            inputVar.buffer().asFloatBuffer().put(inputData1);
            model.run();

            final Variable outputVar = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar.dims())
            );
            final float[] outputBuf = new float[outputVar.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf);
            assertArrayEquals(expectedOutput1, outputBuf);

            // rewrite the internal buffer and run again
            inputVar.buffer().asFloatBuffer().put(inputData2);
            model.run();

            final Variable outputVar2 = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar2.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar2.dims())
            );
            final float[] outputBuf2 = new float[outputVar2.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf2);
            assertArrayEquals(expectedOutput2, outputBuf2);
        }
    }

    @Test
    public void buildAndRunModelIfInputIsDirectBuffer() throws Exception {
        // [[0, 0], [0, 1], [1, 0], [1, 1]] -> [[0], [0], [0], [1]]
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final int inputLen = batchSize * inputDim;
        final ByteBuffer inputDataBuf =
                ByteBuffer.allocateDirect(inputLen * 4).order(ByteOrder.nativeOrder()); // test case
        final float[] inputData1 = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final float[] inputData2 = new float[] {1f, 1f, 1f, 0f, 0f, 1f, 0f, 0f};
        final int outputDim = 1;
        final float[] expectedOutput1 = new float[] {0f, 0f, 0f, 1f};
        final float[] expectedOutput2 = new float[] {1f, 0f, 0f, 0f};

        try (
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder =
                        Model.builder(vpt).attachExternalBuffer("input", inputDataBuf);
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            // you can delete modelData explicitly after building a model
            modelData.close();

            inputDataBuf.asFloatBuffer().put(inputData1);
            model.run();

            final Variable inputVar = model.variable("input");
            assertAll("input variable",
                    () -> assertEquals(DType.FLOAT, inputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, inputDim}, inputVar.dims())
            );
            final int[] inputDims = inputVar.dims();
            final float[] inputBuf = new float[inputDims[0] * inputDims[1]];
            inputVar.buffer().asFloatBuffer().get(inputBuf);
            assertArrayEquals(inputData1, inputBuf);

            final Variable outputVar = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar.dims())
            );
            final float[] outputBuf = new float[outputVar.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf);
            assertArrayEquals(expectedOutput1, outputBuf);

            // rewrite the direct buffer and run again
            inputDataBuf.asFloatBuffer().put(inputData2);
            model.run();

            final Variable outputVar2 = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar2.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar2.dims())
            );
            final float[] outputBuf2 = new float[outputVar2.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf2);
            assertArrayEquals(expectedOutput2, outputBuf2);
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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder =
                        Model.builder(vpt).attachExternalBuffer("input", inputDataBuf);
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            // you can delete modelData explicitly after building a model
            modelData.close();

            model.run();

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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder =
                        Model.builder(vpt).attachExternalBuffer("input", readOnlyInputDataBuf);
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            // you can delete modelData explicitly after building a model
            modelData.close();

            model.run();

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
                ModelData modelData = ModelData.fromOnnxFile(path);
                VariableProfileTableBuilder vptBuilder = VariableProfileTable.builder()
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputProfile("output", DType.FLOAT);
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder =
                        Model.builder(vpt).attachExternalBuffer("input", inputData);
                Model model = modelBuilder.build(modelData, "mkldnn", "")
        ) {
            // you can delete modelData explicitly after building a model
            modelData.close();

            model.run();

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
