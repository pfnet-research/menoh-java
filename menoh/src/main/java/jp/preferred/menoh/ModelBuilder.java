package jp.preferred.menoh;

import static jp.preferred.menoh.BufferUtils.copyToNativeMemory;
import static jp.preferred.menoh.BufferUtils.toDirectByteBuffer;
import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.ByteBuffer;

public class ModelBuilder implements AutoCloseable {
    private Pointer handle;

    private ModelBuilder(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model_builder(handle);
                handle = Pointer.NULL;
            }
        }
    }

    public static ModelBuilder make(VariableProfileTable vpt) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_builder(vpt.nativeHandle(), ref));

        return new ModelBuilder(ref.getValue());
    }

    public void attach(String variableName, float[] values) throws MenohException {
        attach(variableName, values, 0, values.length);
    }

    public void attach(String variableName, float[] values, int offset, int length) throws MenohException {
        attach(variableName, toDirectByteBuffer(values, offset, length));
    }

    public void attach(String variableName, ByteBuffer buffer) throws MenohException {
        final Pointer bufferHandle = copyToNativeMemory(buffer);
        checkError(MenohNative.INSTANCE.menoh_model_builder_attach_external_buffer(handle, variableName, bufferHandle));
    }

    public Model build(ModelData modelData, String backendName, String backendConfig)
            throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_build_model(
                this.handle, modelData.nativeHandle(), backendName, backendConfig, ref));

        return new Model(ref.getValue());
    }
}
