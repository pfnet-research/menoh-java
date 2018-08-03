package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

/**
 * <p>A table which holds {@link VariableProfile}s for the {@link Model}. It will be built by
 * {@link VariableProfileTableBuilder}.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
 */
public class VariableProfileTable implements AutoCloseable {
    private Pointer handle;

    VariableProfileTable(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_variable_profile_table(handle);
                handle = Pointer.NULL;
            }
        }
    }

    /**
     * Creates a {@link VariableProfileTableBuilder}.
     */
    public static VariableProfileTableBuilder builder() throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_variable_profile_table_builder(ref));

        return new VariableProfileTableBuilder(ref.getValue());
    }

    /**
     * A {@link VariableProfile} with the specified name.
     */
    public VariableProfile variableProfile(String variableName) throws MenohException {
        final IntByReference dtype = new IntByReference(0);
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dtype(handle, variableName, dtype));

        final IntByReference dimsSize = new IntByReference();
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dims_size(handle, variableName, dimsSize));

        final int[] dims = new int[dimsSize.getValue()];

        for (int i = 0; i < dimsSize.getValue(); ++i) {
            final IntByReference d = new IntByReference();
            checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dims_at(handle, variableName, i, d));
            dims[i] = d.getValue();
        }

        return new VariableProfile(DType.valueOf(dtype.getValue()), dims);
    }
}
