package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

/**
 * A model data.
 */
public class ModelData implements AutoCloseable {
    private Pointer handle;

    private ModelData(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    /**
     * Optimizes this model data.
     *
     * @return this object
     */
    public ModelData optimize(VariableProfileTable vpt) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_data_optimize(handle, vpt.nativeHandle()));

        return this;
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

    /**
     * Loads an ONNX model from the specified file.
     */
    public static ModelData fromOnnxFile(String onnxModelPath) throws MenohException {
        final PointerByReference handle = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_data_from_onnx(onnxModelPath, handle));

        return new ModelData(handle.getValue());
    }
}
