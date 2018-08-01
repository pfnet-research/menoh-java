package jp.preferred.menoh;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * A convenient wrapper for building and running {@link Model}.
 */
public class ModelRunner implements AutoCloseable {
    private final ModelRunnerBuilder builder;

    private final Model model;

    private static final String DEFAULT_BACKEND_NAME = "mkldnn";

    private static final String DEFAULT_BACKEND_CONFIG = "";

    ModelRunner(ModelRunnerBuilder builder, Model model) {
        this.builder = builder;
        this.model = model;
    }

    ModelRunnerBuilder builder() {
        return this.builder;
    }

    /**
     * Returns the underlying {@link Model}.
     */
    public Model model() {
        return this.model;
    }

    @Override
    public void close() {
        model.close();
        builder.close();
    }

    public static ModelRunnerBuilder fromOnnxFile(String path) {
        ModelData modelData = null;
        VariableProfileTableBuilder vptBuilder = null;
        try {
            modelData = ModelData.fromOnnxFile(path);
            vptBuilder = VariableProfileTable.builder();

            return new ModelRunnerBuilder(
                    modelData,
                    vptBuilder,
                    DEFAULT_BACKEND_NAME,
                    DEFAULT_BACKEND_CONFIG,
                    new HashMap<String, ByteBuffer>());
        } catch (Throwable t) {
            if (modelData != null) {
                modelData.close();
            }
            if (vptBuilder != null) {
                vptBuilder.close();
            }

            throw t;
        }
    }

    /**
     * <p>Run this model after copying a non-empty array to the specified variable.</p>
     *
     * @param name the name of the input variable
     * @param values the values to be copied to the input variable
     * @return the output variables of the model
     */
    public void run(String name, float[] values) {
        run(name, values, 0, values.length);
    }

    /**
     * <p>Run this model after copying a non-empty array to the specified variable. It copies the content
     * ranging from <code>offset</code> to <code>(offset + length - 1)</code>.</p>
     *
     * @param name the name of the input variable
     * @param values the values to be copied to the input variable
     * @return the output variables of the model
     */
    public void run(String name, float[] values, int offset, int length) {
        final ByteBuffer buffer = ByteBuffer.allocateDirect(length * 4).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(values, offset, length);

        run(name, buffer);
    }

    /**
     * <p>Run this model after copying a non-empty buffer to the specified variable. It copies the content
     * ranging from <code>position()</code> to <code>(limit() - 1)</code> without changing them, even if
     * the <code>buffer</code> is direct unlike {@link ModelRunnerBuilder#attach(String, ByteBuffer)}.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param name the name of the input variable
     * @param buffer the buffer to be copied to the input variable
     * @return the output variables of the model
     */
    public void run(String name, ByteBuffer buffer) {
        run(Collections.singletonMap(name, buffer));
    }

    /**
     * <p>Run this model after copying non-empty buffers to the specified variables. It copies the content
     * ranging from <code>position()</code> to <code>(limit() - 1)</code> without changing them, even if
     * the <code>buffer</code> is direct unlike {@link ModelRunnerBuilder#attach(String, ByteBuffer)}.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffers the buffers to be copied to the variables
     * @return the output variables of the model
     */
    public void run(final Map<String, ByteBuffer> buffers) {
        assignToVariables(buffers);
        model.run();
    }

    /**
     * Run this model.
     */
    public void run() {
        model.run();
    }

    /**
     * Assign data to the variables of the model.
     */
    private void assignToVariables(final Map<String, ByteBuffer> data) {
        for (Map.Entry<String, ByteBuffer> e : data.entrySet()) {
            final String name = e.getKey();
            final ByteBuffer dataBuf = e.getValue();
            final long dataLen = dataBuf.remaining();

            final Variable v = model.variable(name);
            final long varLen = v.bufferLength();

            if (varLen < dataLen) {
                throw new MenohRunnerException(String.format(
                        "The data with length > %d can't be assigned to the variable `%s`.", varLen, name));
            }

            final ByteBuffer varBuf = v.buffer();
            varBuf.clear();
            varBuf.put(dataBuf.duplicate()).rewind();
        }
    }
}
