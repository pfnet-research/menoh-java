package jp.preferred.menoh;

// CHECKSTYLE:OFF
import static jp.preferred.menoh.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import java.io.InputStream;

import org.apache.commons.io.IOUtils;
import org.junit.jupiter.api.Test;

public class ModelRunnerTest {
    @Test
    public void runModelRunner() throws Exception {
        final String path = getResourceFilePath("models/and_op.onnx"); // loaded from file
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData1 = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};
        final float[] inputData2 = new float[] {1f, 1f, 1f, 0f, 0f, 1f, 0f, 0f};
        final int outputDim = 1;
        final float[] expectedOutput1 = new float[] {0f, 0f, 0f, 1f};
        final float[] expectedOutput2 = new float[] {1f, 0f, 0f, 0f};

        try (
                ModelRunnerBuilder builder = ModelRunner
                        .fromOnnxFile(path) // loaded from a file
                        .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                        .addOutputName("output")
                        .attachExternalBuffer("input", inputData1)
                        .backendName("mkldnn")
                        .backendConfig("");
                 ModelRunner runner = builder.build()
        ) {
            assertAll("model data in builder",
                    () -> assertNotNull(builder.modelData()),
                    () -> assertNotNull(builder.modelData().nativeHandle()),
                    () -> assertNull(builder.modelData().nativeDataPointer()) // loaded from file directly
            );
            assertAll("vpt builder in builder",
                    () -> assertNotNull(builder.vptBuilder()),
                    () -> assertNotNull(builder.vptBuilder().nativeHandle())
            );
            assertAll("backend config in builder",
                    () -> assertEquals("mkldnn", builder.backendName()),
                    () -> assertEquals("", builder.backendConfig())
            );
            assertAll("attached external buffers in builder",
                    () -> assertNotNull(builder.externalBuffers()),
                    () -> assertNotNull(builder.externalBuffers().get("input"))
            );

            assertAll("model in runner",
                    () -> assertNotNull(runner.model()),
                    () -> assertNotNull(runner.model().nativeHandle())
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
        final InputStream in = getClass().getClassLoader().getResourceAsStream("models/and_op.onnx");
        final byte[] data = IOUtils.toByteArray(in);
        final int batchSize = 4;
        final int inputDim = 2;
        final float[] inputData = new float[] {0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f};

        final ModelRunnerBuilder builder = ModelRunner
                .fromOnnx(data) // loaded from memory
                .addInputProfile("input", DType.FLOAT, new int[] {batchSize, inputDim})
                .addOutputName("output")
                .attachExternalBuffer("input", inputData)
                .backendName("mkldnn")
                .backendConfig("");
        final ModelRunner runner = builder.build();
        try {
            assertNotNull(builder);
            assertAll("model data in builder",
                    () -> assertNotNull(builder.modelData()),
                    () -> assertNotNull(builder.modelData().nativeHandle()),
                    () -> assertNotNull(builder.modelData().nativeDataPointer()),
                    () -> assertFalse(builder.modelData().nativeDataPointer().isDisposed())
            );
            assertAll("vpt builder in builder",
                    () -> assertNotNull(builder.vptBuilder()),
                    () -> assertNotNull(builder.vptBuilder().nativeHandle())
            );
            assertAll("backend config in builder",
                    () -> assertEquals("mkldnn", builder.backendName()),
                    () -> assertEquals("", builder.backendConfig())
            );
            assertAll("attached external buffers in builder",
                    () -> assertNotNull(builder.externalBuffers()),
                    () -> assertNotNull(builder.externalBuffers().get("input"))
            );

            assertNotNull(runner);
            assertAll("model in runner",
                    () -> assertNotNull(runner.model()),
                    () -> assertNotNull(runner.model().nativeHandle())
            );
        } finally {
            builder.close();
            assertAll("model data in builder",
                    () -> assertNotNull(builder.modelData()),
                    () -> assertNull(builder.modelData().nativeHandle()),
                    () -> assertNotNull(builder.modelData().nativeDataPointer()),
                    () -> assertTrue(builder.modelData().nativeDataPointer().isDisposed())
            );
            assertAll("vpt builder in builder",
                    () -> assertNotNull(builder.vptBuilder()),
                    () -> assertNull(builder.vptBuilder().nativeHandle())
            );
            assertAll("attached external buffers in builder",
                    () -> assertNotNull(builder.externalBuffers()),
                    () -> assertTrue(builder.externalBuffers().isEmpty(),
                            "externalBuffers should be empty")
            );

            runner.close();
            assertAll("model",
                    () -> assertNotNull(runner.model()),
                    () -> assertNull(runner.model().nativeHandle())
            );

            // close() is an idempotent operation
            builder.close();
            runner.close();
        }
    }
}
