package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

import java.util.List;

/**
 * <p>A representation of the model. It will be built by {@link ModelBuilder}.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
 */
public class Model implements AutoCloseable {
    private Pointer handle;

    /**
     * A reference to the pointers to prevent them from getting garbage collected.
     */
    private final List<Pointer> externalBuffers;

    Model(Pointer handle, List<Pointer> externalBuffers) {
        this.handle = handle;
        this.externalBuffers = externalBuffers;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    List<Pointer> externalBuffers() {
        return this.externalBuffers;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_model(handle);
                handle = Pointer.NULL;
                externalBuffers.clear();
            }
        }
    }

    /**
     * Creates a {@link ModelBuilder}.
     */
    public static ModelBuilder builder(VariableProfileTable vpt) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_builder(vpt.nativeHandle(), ref));

        return new ModelBuilder(ref.getValue());
    }

    /**
     * Returns a {@link Variable} with the specified name.
     */
    public Variable variable(String variableName) throws MenohException {
        final IntByReference dtype = new IntByReference();

        checkError(MenohNative.INSTANCE.menoh_model_get_variable_dtype(handle, variableName, dtype));

        final IntByReference dimsSize = new IntByReference();
        checkError(MenohNative.INSTANCE.menoh_model_get_variable_dims_size(handle, variableName, dimsSize));

        final int[] dims = new int[dimsSize.getValue()];

        for (int i = 0; i < dimsSize.getValue(); ++i) {
            final IntByReference d = new IntByReference();
            checkError(MenohNative.INSTANCE.menoh_model_get_variable_dims_at(handle, variableName, i, d));
            dims[i] = d.getValue();
        }

        final PointerByReference buffer = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_model_get_variable_buffer_handle(handle, variableName, buffer));

        return new Variable(DType.valueOf(dtype.getValue()), dims, buffer.getValue());
    }

    /**
     * Run this model.
     */
    public void run() throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_model_run(handle));
    }
}
