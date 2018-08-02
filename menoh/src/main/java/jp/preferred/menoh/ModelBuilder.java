package jp.preferred.menoh;

import static jp.preferred.menoh.BufferUtils.copyToNativeMemory;
import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * A builder object for {@link Model}.
 */
public class ModelBuilder implements AutoCloseable {
    private Pointer handle;

    /**
     * A reference to the pointers to prevent them from getting garbage collected.
     */
    private final List<Pointer> externalBuffers = new ArrayList<>();

    ModelBuilder(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    List<Pointer> externalBuffers() {
        return this.externalBuffers;
    }

    /**
     * Releases the model builder.
     */
    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model_builder(handle);
                handle = Pointer.NULL;
                externalBuffers.clear();
            }
        }
    }

    /**
     * <p>Attaches a non-empty external buffer to the specified variable.</p>
     *
     * <p>If the specified <code>buffer</code> is direct, it will be attached to the model directly without
     * copying. Otherwise, it copies the content to a newly allocated buffer in the native heap ranging from
     * <code>position()</code> to <code>(limit() - 1)</code> without changing its position.</p>
     *
     * <p>The attached buffer can be accessed through {@link Model#variable(String)}.</p>
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
    public ModelBuilder attach(String variableName, ByteBuffer buffer) throws MenohException {
        final Pointer bufferHandle = copyToNativeMemory(buffer);
        synchronized (this) {
            externalBuffers.add(bufferHandle);
        }

        return attachImpl(variableName, bufferHandle);
    }

    /**
     * <p>Attaches a non-empty external buffer to the specified variable. It also copies the content of the
     * <code>values</code> to a newly allocated buffer in the native heap.</p>
     *
     * <p>The buffer can be accessed through {@link Model#variable(String)}.</p>
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    public ModelBuilder attach(String variableName, float[] values) throws MenohException {
        return attach(variableName, values, 0, values.length);
    }

    /**
     * <p>Attaches a non-empty external buffer to the specified variable. It also copies the content of the
     * <code>values</code> to a newly allocated buffer in the native heap ranging from <code>offset</code>
     * to <code>(offset + length - 1)</code>.</p>
     *
     * <p>The buffer can be accessed through {@link Model#variable(String)}.</p>
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    public ModelBuilder attach(String variableName, float[] values, int offset, int length) throws MenohException {
        final Pointer bufferHandle = copyToNativeMemory(values, offset, length);
        synchronized (this) {
            externalBuffers.add(bufferHandle);
        }

        return attachImpl(variableName, bufferHandle);
    }

    private ModelBuilder attachImpl(String variableName, Pointer bufferHandle) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_builder_attach_external_buffer(
                handle, variableName, bufferHandle));

        return this;
    }

    /**
     * <p>Builds a {@link Model} to <code>run()</code> by using the specified backend (e.g. "mkldnn").</p>
     *
     * <p>Menoh will allocate a new buffer for input and output variables to which an external buffer is not
     * attached. It can be accessed via {@link Model#variable(String)}.</p>
     */
    public Model build(ModelData modelData, String backendName, String backendConfig) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_build_model(
                this.handle, modelData.nativeHandle(), backendName, backendConfig, ref));

        synchronized (this) {
            return new Model(ref.getValue(), new ArrayList<>(this.externalBuffers));
        }
    }
}
