package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class ModelRunnerTest {
    @Test
    public void runModelRunner() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData1 = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final float[] inputData2 = new float[] {1f, 1f, 1f, 0f, 0f, 1f, 0f, 0f};
        final int outputDim = 1;
        final float[] expectedOutput1 = new float[] {0f, 0f, 0f, 1f};
        final float[] expectedOutput2 = new float[] {1f, 0f, 0f, 0f};

        try (ModelRunner runner = ModelRunner
                .fromOnnxFile(path)
                .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                .addOutputProfile("output", DType.FLOAT)
                .attach("input", inputData1)
                .backendName("mkldnn")
                .backendConfig("")
                .build()) {

            assertAll("model",
                    () -> assertNotNull(runner.model()),
                    () -> assertNotNull(runner.model().nativeHandle())
            );
            assertNotNull(runner.builder());
            assertAll("model data in builder",
                    () -> assertNotNull(runner.builder().modelData()),
                    () -> assertNull(runner.builder().modelData().nativeHandle()) // deleted in build()
            );
            assertAll("vpt builder in builder",
                    () -> assertNotNull(runner.builder().vptBuilder()),
                    () -> assertNotNull(runner.builder().vptBuilder().nativeHandle())
            );
            assertAll("backend config in builder",
                    () -> assertEquals("mkldnn", runner.builder().backendName()),
                    () -> assertEquals("", runner.builder().backendConfig())
            );
            assertAll("attached buffers in builder",
                    () -> assertNotNull(runner.builder().attachedBuffers()),
                    () -> assertNotNull(runner.builder().attachedBuffers().get("input"))
            );

            // run the model
            runner.run();

            final Model model = runner.model();
            final Variable outputVar = model.variable("output");
            assertAll("output variable",
                    () -> assertEquals(DType.FLOAT, outputVar.dtype()),
                    () -> assertArrayEquals(new int[] {batchSize, outputDim}, outputVar.dims())
            );
            final float[] outputBuf = new float[outputVar.dims()[0]];
            outputVar.buffer().asFloatBuffer().get(outputBuf);
            assertArrayEquals(expectedOutput1, outputBuf);

            // rewrite the internal buffer and run again
            runner.run("input", inputData2);

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
    public void closeModelRunner() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx");
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        final ModelRunner runner = ModelRunner
                .fromOnnxFile(path)
                .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                .addOutputProfile("output", DType.FLOAT)
                .attach("input", inputData)
                .backendName("mkldnn")
                .backendConfig("")
                .build();
        try {
            assertAll("model",
                    () -> assertNotNull(runner.model()),
                    () -> assertNotNull(runner.model().nativeHandle())
            );
            assertNotNull(runner.builder());
            assertAll("model data in builder",
                    () -> assertNotNull(runner.builder().modelData()),
                    () -> assertNull(runner.builder().modelData().nativeHandle()) // deleted in build()
            );
            assertAll("vpt builder in builder",
                    () -> assertNotNull(runner.builder().vptBuilder()),
                    () -> assertNotNull(runner.builder().vptBuilder().nativeHandle())
            );
            assertAll("backend config in builder",
                    () -> assertEquals("mkldnn", runner.builder().backendName()),
                    () -> assertEquals("", runner.builder().backendConfig())
            );
            assertAll("attached buffers in builder",
                    () -> assertNotNull(runner.builder().attachedBuffers()),
                    () -> assertNotNull(runner.builder().attachedBuffers().get("input"))
            );
        } finally {
            runner.close();
            assertAll("model",
                    () -> assertNotNull(runner.model()),
                    () -> assertNull(runner.model().nativeHandle())
            );
            assertNotNull(runner.builder());
            assertAll("vpt builder in builder",
                    () -> assertNotNull(runner.builder().vptBuilder()),
                    () -> assertNull(runner.builder().vptBuilder().nativeHandle())
            );
            assertAll("attached buffers in builder",
                    () -> assertNotNull(runner.builder().attachedBuffers()),
                    () -> assertTrue(runner.builder().attachedBuffers().isEmpty(),
                            "attachedBuffers should be empty")
            );

            // close() is an idempotent operation
            runner.close();
        }
    }
}
