package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

public class VariableProfileTableBuilder implements AutoCloseable {
    private Pointer handle;

    private VariableProfileTableBuilder(Pointer handle) {
        this.handle = handle;
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

    public static VariableProfileTableBuilder make() throws MenohException {
        PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_variable_profile_table_builder(ref));

        return new VariableProfileTableBuilder(ref.getValue());
    }

    public void addInputProfile(String name, DType dtype, int[] dims) throws MenohException {
        if (dims.length == 2) {
            checkError(
                    MenohNative.INSTANCE.menoh_variable_profile_table_builder_add_input_profile_dims_2(
                            handle, name, dtype.getId(), dims[0], dims[1]));
        } else if (dims.length == 4) {
            checkError(
                    MenohNative.INSTANCE.menoh_variable_profile_table_builder_add_input_profile_dims_4(
                            handle, name, dtype.getId(), dims[0], dims[1], dims[2], dims[3]));
        } else {
            throw new MenohException(
                    ErrorCode.UNDEFINED,
                    String.format("%s has an invalid dims size: %d (it must be 2 or 4)", name, dims.length));
        }
    }

    public void addOutputProfile(String name, DType dtype) throws MenohException {
        checkError(MenohNative.INSTANCE.menoh_variable_profile_table_builder_add_output_profile(
                handle, name, dtype.getId()));
    }

    public VariableProfileTable buildVariableProfileTable(ModelData modelData) throws MenohException {
        PointerByReference ref = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_build_variable_profile_table(
                this.handle, modelData.getNativeHandle(), ref));

        return new VariableProfileTable(ref.getValue());
    }
}
