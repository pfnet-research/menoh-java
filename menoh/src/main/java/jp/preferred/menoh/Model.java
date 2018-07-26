package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

/**
 * It is created by {@link ModelBuilder}.
 */
public class Model implements AutoCloseable {
    private Pointer handle;

    /**
     * @param handle a pointer to a native <code>menoh_model</code>
     */
    Model(Pointer handle) {
        this.handle = handle;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model(handle);
                handle = Pointer.NULL;
            }
        }
    }

    public Variable getVariable(String name) throws MenohException {
        final IntByReference dtype = new IntByReference();

        checkError(MenohNative.INSTANCE.menoh_model_get_variable_dtype(handle, name, dtype));

        final IntByReference dimsSize = new IntByReference();
        checkError(MenohNative.INSTANCE.menoh_model_get_variable_dims_size(handle, name, dimsSize));

        final int[] dims = new int[dimsSize.getValue()];

        for (int i = 0; i < dimsSize.getValue(); ++i) {
            final IntByReference d = new IntByReference();
            checkError(MenohNative.INSTANCE.menoh_model_get_variable_dims_at(handle, name, i, d));
            dims[i] = d.getValue();
        }

        final PointerByReference buffer = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_model_get_variable_buffer_handle(handle, name, buffer));

        return new Variable(DType.valueOf(dtype.getValue()), dims, buffer.getValue());
    }

    public void run() throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_run(handle));
    }
}
