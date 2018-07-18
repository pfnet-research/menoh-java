package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class ModelData implements AutoCloseable {
    private Pointer handle;

    /**
     * @param handle a pointer to the native <code>menoh_model_data</code>
     */
    private ModelData(Pointer handle) {
        this.handle = handle;
    }

    Pointer getNativeHandle() {
        return this.handle;
    }

    public void optimize(VariableProfileTable vpt) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_data_optimize(handle, vpt.getNativeHandle()));
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model_data(handle);
                handle = Pointer.NULL;
            }
        }
    }

    public static ModelData makeModelDataFromOnnx(String onnxModelPath) throws MenohException {
        PointerByReference handle = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_data_from_onnx(onnxModelPath, handle));

        return new ModelData(handle.getValue());
    }
}
