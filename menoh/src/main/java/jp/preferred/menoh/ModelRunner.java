package jp.preferred.menoh;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * <p>A convenient wrapper for building and running {@link Model}.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
 */
public class ModelRunner implements AutoCloseable {
    private final Model model;

    private static final String DEFAULT_BACKEND_NAME = "mkldnn";

    private static final String DEFAULT_BACKEND_CONFIG = "";

    ModelRunner(Model model) {
        this.model = model;
    }

    /**
     * Returns the underlying {@link Model}.
     */
    Model model() {
        return this.model;
    }

    @Override
    public void close() {
        model.close();
    }

    /**
     * <p>Loads an ONNX model from the specified byte array.</p>
     */
    public static ModelRunnerBuilder fromOnnx(byte[] data) {
        return fromOnnx(data, 0, data.length);
    }

    /**
     * <p>Loads an ONNX model from the specified byte array.</p>
     */
    public static ModelRunnerBuilder fromOnnx(byte[] data, int offset, int size) {
        ModelData modelData = null;
        VariableProfileTableBuilder vptBuilder = null;
        try {
            modelData = ModelData.fromOnnx(data, offset, size);
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
     * <p>Loads an ONNX model from the specified byte buffer.</p>
     *
     * <p>If the specified <code>buffer</code> is direct, it will be directly used without copying.
     * Otherwise, it copies the content to a newly allocated buffer in the native heap ranging from
     * <code>position()</code> to <code>(limit() - 1)</code> without changing its position.</p>
     */
    public static ModelRunnerBuilder fromOnnx(ByteBuffer data) {
        ModelData modelData = null;
        VariableProfileTableBuilder vptBuilder = null;
        try {
            modelData = ModelData.fromOnnx(data);
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
     * <p>Loads an ONNX model from the specified file.</p>
     */
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
     * Returns a {@link Variable} with the specified name.
     */
    public Variable variable(String variableName) throws MenohException {
        return model.variable(variableName);
    }

    /**
     * <p>Run this model after assigning a non-empty array to the specified variable.</p>
     *
     * @param name the name of the input variable
     * @param values the values to be copied to the input variable
     */
    public void run(String name, float[] values) {
        run(name, values, 0, values.length);
    }

    /**
     * <p>Run this model after assigning a non-empty array to the specified variable. It copies the content
     * ranging from <code>offset</code> to <code>(offset + length - 1)</code>.</p>
     *
     * @param name the name of the input variable
     * @param values the values to be copied to the input variable
     */
    public void run(String name, float[] values, int offset, int length) {
        final ByteBuffer buffer = ByteBuffer.allocateDirect(length * 4).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(values, offset, length);

        run(name, buffer);
    }

    /**
     * <p>Run this model after assigning a non-empty buffer to the specified variable. It copies the content
     * ranging from <code>position()</code> to <code>(limit() - 1)</code> without changing them, even if
     * the <code>buffer</code> is direct unlike {@link ModelRunnerBuilder#attachExternalBuffer(String, ByteBuffer)}.
     * </p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param name the name of the input variable
     * @param buffer the buffer to be copied to the input variable
     */
    public void run(String name, ByteBuffer buffer) {
        run(Collections.singletonMap(name, buffer));
    }

    /**
     * <p>Run this model after assigning non-empty buffers to the specified variables. It copies the content
     * ranging from <code>position()</code> to <code>(limit() - 1)</code> without changing them, even if
     * the <code>buffer</code> is direct unlike {@link ModelRunnerBuilder#attachExternalBuffer(String, ByteBuffer)}.
     * </p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffers the buffers to be copied to the variables
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
     * Assign data to the variables in the model.
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
