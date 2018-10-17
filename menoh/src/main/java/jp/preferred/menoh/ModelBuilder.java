package jp.preferred.menoh;

import static jp.preferred.menoh.BufferUtils.convertToNativeMemory;
import static jp.preferred.menoh.BufferUtils.copyToNativeMemory;
import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>A builder object for {@link Model}.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
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
     * @deprecated This API is not useful at this point in Java. Use {@link Model#variable(String)} instead.
     * We will redesign it in the future.
     *
     * @param variableName the name of the variable
     * @param buffer the byte buffer from which to copy
     * @return this object
     *
     * @throws IllegalArgumentException if <code>buffer</code> is null or empty
     */
    @Deprecated
    public ModelBuilder attachExternalBuffer(String variableName, ByteBuffer buffer) throws MenohException {
        if (buffer == null || buffer.remaining() <= 0) {
            throw new IllegalArgumentException("data must not be null or empty");
        }

        final Pointer bufferHandle;
        if (buffer.isDirect()) {
            bufferHandle = convertToNativeMemory(buffer);
        } else {
            // TODO: free the allocated memory explicitly after all models and its builder is closed
            // (JVM will invoke Memory#dispose() after bufferHandle is GCed)
            bufferHandle = copyToNativeMemory(buffer);
        }

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
     * @deprecated This API is not useful at this point in Java. Use {@link Model#variable(String)} instead.
     * We will redesign it in the future.
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    @Deprecated
    public ModelBuilder attachExternalBuffer(String variableName, float[] values) throws MenohException {
        return attachExternalBuffer(variableName, values, 0, values.length);
    }

    /**
     * <p>Attaches a non-empty external buffer to the specified variable. It also copies the content of the
     * <code>values</code> to a newly allocated buffer in the native heap ranging from <code>offset</code>
     * to <code>(offset + length - 1)</code>.</p>
     *
     * <p>The buffer can be accessed through {@link Model#variable(String)}.</p>
     *
     * @deprecated This API is not useful at this point in Java. Use {@link Model#variable(String)} instead.
     * We will redesign it in the future.
     *
     * @param variableName the name of the variable
     * @param values the byte buffer from which to copy
     * @param offset the array index from which to start copying
     * @param length the number of elements from <code>values</code> that must be copied
     * @return this object
     *
     * @throws IllegalArgumentException if <code>values</code> is null or empty
     */
    @Deprecated
    public ModelBuilder attachExternalBuffer(
            String variableName, float[] values, int offset, int length) throws MenohException {
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
