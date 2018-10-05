package jp.preferred.menoh;

import static jp.preferred.menoh.BufferUtils.copyToNativeMemory;
import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

/**
 * <p>A builder object for {@link VariableProfileTable}.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
 */
public class VariableProfileTableBuilder implements AutoCloseable {
    private Pointer handle;

    VariableProfileTableBuilder(Pointer handle) {
        this.handle = handle;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (handle != Pointer.NULL) {
                MenohNative.INSTANCE.menoh_delete_variable_profile_table_builder(handle);
                handle = Pointer.NULL;
            }
        }
    }

    /**
     * Adds an input profile to configure the specified variable in the model.
     *
     * @return this object
     */
    public VariableProfileTableBuilder addInputProfile(String name, DType dtype, int[] dims) throws MenohException {
        final Pointer dimsPtr = copyToNativeMemory(dims, 0, dims.length);
        checkError(
                MenohNative.INSTANCE.menoh_variable_profile_table_builder_add_input_profile(
                        handle, name, dtype.getId(), dims.length, dimsPtr));

        return this;
    }

    /**
     * Adds an output name to configure the specified variable in the model.
     *
     * @return this object
     */
    public VariableProfileTableBuilder addOutputName(String name) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_builder_add_output_name(handle, name));

        return this;
    }

    /**
     * Adds an output profile to configure the specified variable in the model.
     *
     * @deprecated Use {@link #addOutputName(String)} instead
     *
     * @return this object
     */
    @Deprecated
    public VariableProfileTableBuilder addOutputProfile(String name, DType dtyp) throws MenohException {
        return addOutputName(name);
    }

    /**
     * Builds a {@link VariableProfileTable}. It is used for making {@link ModelBuilder}.
     */
    public VariableProfileTable build(ModelData modelData) throws MenohException {
        final PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_build_variable_profile_table(
                this.handle, modelData.nativeHandle(), ref));

        return new VariableProfileTable(ref.getValue());
    }
}
