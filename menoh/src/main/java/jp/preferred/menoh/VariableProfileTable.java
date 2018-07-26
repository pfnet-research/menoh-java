package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;

/**
 * It is created by {@link VariableProfileTableBuilder}.
 */
public class VariableProfileTable implements AutoCloseable {
    private Pointer handle;

    /**
     * @param handle a pointer to a native <code>menoh_variable_profile_table</code>
     */
    VariableProfileTable(Pointer handle) {
        this.handle = handle;
    }

    Pointer getNativeHandle() {
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

    public VariableProfile getVariableProfile(String name) throws MenohException {
        final IntByReference dtype = new IntByReference(0);
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dtype(handle, name, dtype));

        final IntByReference dimsSize = new IntByReference();
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dims_size(handle, name, dimsSize));

        final int[] dims = new int[dimsSize.getValue()];

        for (int i = 0; i < dimsSize.getValue(); ++i) {
            final IntByReference d = new IntByReference();
            checkError(MenohNative.INSTANCE.menoh_variable_profile_table_get_dims_at(handle, name, i, d));
            dims[i] = d.getValue();
        }

        return new VariableProfile(DType.valueOf(dtype.getValue()), dims);
    }
}
