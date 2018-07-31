package jp.preferred.menoh;

import static jp.preferred.menoh.BufferUtils.copyToNativeMemory;
import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

public class ModelBuilder implements AutoCloseable {
    private Pointer handle;

    /**
     * A reference to the pointers to prevent them from getting garbage collected.
     */
    private final List<Pointer> attachedBuffers = new ArrayList<>();

    private ModelBuilder(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    List<Pointer> attachedBuffers() {
        return this.attachedBuffers;
    }

    /**
     * Release the model builder.
     */
    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model_builder(handle);
                handle = Pointer.NULL;
                attachedBuffers.clear();
            }
        }
    }

    /**
     * Make a {@link ModelBuilder} which can <code>attach()</code> a buffer to a variable of the model.
     */
    public static ModelBuilder make(VariableProfileTable vpt) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_builder(vpt.nativeHandle(), ref));

        return new ModelBuilder(ref.getValue());
    }

    /**
     * <p>Attach a buffer to the specified variable. It copies the content of the buffer to an allocated memory
     * in the native heap ranging from <code>position()</code> to <code>(limit() - 1)</code> without changing them,
     * except for a direct buffer.</p>
     *
     * <p>Note that the <code>order()</code> of the buffer should be {@link ByteOrder#nativeOrder()} because
     * the native byte order of your platform may differ from JVM.</p>
     *
     * @param buffer the byte buffer from which to copy
     */
    public void attach(String variableName, ByteBuffer buffer) throws MenohException {
        final Pointer bufferHandle = copyToNativeMemory(buffer);
        synchronized (this) {
            attachedBuffers.add(bufferHandle);
        }

        attachImpl(variableName, bufferHandle);
    }

    /**
     * <p>Attach an array to the specified variable. It copies the content of the array to an allocated memory
     * in the native heap.</p>
     *
     * @param values the byte buffer from which to copy
     */
    public void attach(String variableName, float[] values) throws MenohException {
        attach(variableName, values, 0, values.length);
    }

    /**
     * <p>Attach an array to the specified variable. It copies the content of the array to an allocated memory
     * in the native heap ranging from <code>offset</code> to <code>(offset + length - 1)</code>.</p>
     *
     * @param values the byte buffer from which to copy
     */
    public void attach(String variableName, float[] values, int offset, int length) throws MenohException {
        final Pointer bufferHandle = copyToNativeMemory(values, offset, length);
        synchronized (this) {
            attachedBuffers.add(bufferHandle);
        }

        attachImpl(variableName, bufferHandle);
    }

    private void attachImpl(String variableName, Pointer bufferHandle) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_builder_attach_external_buffer(
                handle, variableName, bufferHandle));
    }

    /**
     * Build a {@link Model} to <code>run()</code> based on a {@link ModelData} and the attached buffers
     * by using the specified backend.
     */
    public Model build(ModelData modelData, String backendName, String backendConfig) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_build_model(
                this.handle, modelData.nativeHandle(), backendName, backendConfig, ref));

        synchronized (this) {
            return new Model(ref.getValue(), new ArrayList<>(this.attachedBuffers));
        }
    }
}
