package jp.preferred.menoh;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Map;

/**
 * A builder object for {@link ModelRunner}.
 */
public class ModelRunnerBuilder implements AutoCloseable {
    private final ModelData modelData;

    private final VariableProfileTableBuilder vptBuilder;

    private String backendName;

    private String backendConfig;

    private final Map<String, ByteBuffer> attachedBuffers;

    ModelRunnerBuilder(
            ModelData modelData,
            VariableProfileTableBuilder vptBuilder,
            String backendName,
            String backendConfig,
            Map<String, ByteBuffer> attachedBuffers) {
        this.modelData = modelData;
        this.vptBuilder = vptBuilder;
        this.backendName = backendName;
        this.backendConfig = backendConfig;
        this.attachedBuffers = attachedBuffers;
    }

    ModelData modelData() {
        return this.modelData;
    }

    VariableProfileTableBuilder vptBuilder() {
        return this.vptBuilder;
    }

    String backendName() {
        return this.backendName;
    }

    public ModelRunnerBuilder backendName(String backendName) {
        this.backendName = backendName;
        return this;
    }

    String backendConfig() {
        return this.backendConfig;
    }

    public ModelRunnerBuilder backendConfig(String backendConfig) {
        this.backendConfig = backendConfig;
        return this;
    }

    Map<String, ByteBuffer> attachedBuffers() {
        return this.attachedBuffers;
    }

    @Override
    public void close() {
        vptBuilder.close();
        modelData.close();

        // allow the attached buffers to GC its allocated memory
        attachedBuffers.clear();
    }

    /**
     * Adds an input profile to configure the specified variable of the model.
     *
     * @return this object
     */
    public ModelRunnerBuilder addInputProfile(String name, DType dtype, int[] dims) {
        vptBuilder.addInputProfile(name, dtype, dims);
        return this;
    }

    /**
     * Adds an output profile to configure the specified variable of the model.
     *
     * @return this object
     */
    public ModelRunnerBuilder addOutputProfile(String name, DType dtype) {
        vptBuilder.addOutputProfile(name, dtype);
        return this;
    }

    /**
     * <p>Attaches a non-empty buffer to the specified variable.</p>
     *
     * <p>If the <code>buffer</code> is direct, it will be attached to the model directly without copying.
     * You can <code>run()</code> the model again and again by updating the attached buffer instead of
     * rebuilding the model.</p>
     *
     * <p>Otherwise, it copies the content of the buffer to an allocated memory in the native heap ranging
     * from <code>position()</code> to <code>(limit() - 1)</code> without changing them.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param variableName the name of the variable
     * @param buffer the byte buffer from which to copy
     * @return this object
     *
     * @throws IllegalArgumentException if <code>buffer</code> is null or empty
     */
    public ModelRunnerBuilder attach(String variableName, ByteBuffer buffer) throws MenohException {
        attachedBuffers.put(variableName, buffer);
        return this;
    }

    /**
     * <p>Attaches a non-empty array to the specified variable. It copies the content of the array to an allocated
     * memory in the native heap.</p>
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    public ModelRunnerBuilder attach(String variableName, float[] values) throws MenohException {
        return attach(variableName, values, 0, values.length);
    }

    /**
     * <p>Attaches a non-empty array to the specified variable. It copies the content of the array to an allocated
     * memory in the native heap ranging from <code>offset</code> to <code>(offset + length - 1)</code>.</p>
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    public ModelRunnerBuilder attach(
            String variableName,
            float[] values,
            int offset,
            int length) throws MenohException {
        // copy the array to a direct buffer
        final ByteBuffer buffer = ByteBuffer.allocateDirect(length * 4).order(ByteOrder.nativeOrder());
        buffer.asFloatBuffer().put(values, offset, length);

        attachedBuffers.put(variableName, buffer);
        return this;
    }

    /**
     * <p>Builds a {@link ModelRunner} to <code>run()</code> by using the specified backend (e.g. "mkldnn").</p>
     *
     * <p>Menoh will allocate a new buffer for input and output variables to which an external buffer is not
     * attached. It can be accessed via {@link Model#variable(String)} in the <code>ModelRunner</code> object.</p>
     */
    public ModelRunner build() {
        try (
                VariableProfileTable vpt = vptBuilder.build(modelData);
                ModelBuilder modelBuilder = Model.builder(vpt)
        ) {
            for (Map.Entry<String, ByteBuffer> e : attachedBuffers.entrySet()) {
                modelBuilder.attach(e.getKey(), e.getValue());
            }

            final Model model = modelBuilder.build(modelData, backendName, backendConfig);
            return new ModelRunner(model);
        }
    }
}
