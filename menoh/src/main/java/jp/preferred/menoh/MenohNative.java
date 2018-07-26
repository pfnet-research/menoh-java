package jp.preferred.menoh;

import com.sun.jna.Library;
import com.sun.jna.Native;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.IntByReference;
import com.sun.jna.ptr.PointerByReference;

// CHECKSTYLE:OFF
interface MenohNative extends Library {
    MenohNative INSTANCE = (MenohNative) Native.loadLibrary("menoh", MenohNative.class);

    String menoh_get_last_error_message();

    int menoh_make_model_data_from_onnx(String onnx_filename, PointerByReference dst_handle);

    void menoh_delete_model_data(Pointer model_data);

    int menoh_model_data_optimize(Pointer model_data, Pointer variable_profile_table);

    int menoh_make_variable_profile_table_builder(PointerByReference dst_handle);

    void menoh_delete_variable_profile_table_builder(Pointer builder);

    int menoh_variable_profile_table_builder_add_input_profile_dims_2(Pointer builder, String name, int dtype, int num, int size);

    int menoh_variable_profile_table_builder_add_input_profile_dims_4(Pointer builder, String name, int dtype, int num, int channel, int height, int width);

    int menoh_variable_profile_table_builder_add_output_profile(Pointer builder, String name, int dtype);

    int menoh_build_variable_profile_table(Pointer builder, Pointer model_data, PointerByReference dst_handle);

    void menoh_delete_variable_profile_table(Pointer variable_profile_table);

    int menoh_variable_profile_table_get_dtype(Pointer variable_profile_table, String variable_name, IntByReference dst_dtype);

    int menoh_variable_profile_table_get_dims_size(Pointer variable_profile_table, String variable_name, IntByReference dst_size);

    int menoh_variable_profile_table_get_dims_at(Pointer variable_profile_table, String variable_name, int index, IntByReference dst_size);

    int menoh_make_model_builder(Pointer variable_profile_table, PointerByReference dst_handle);

    void menoh_delete_model_builder(Pointer model_builder);

    int menoh_model_builder_attach_external_buffer(Pointer builder, String variable_name, Pointer buffer_handle);

    int menoh_build_model(Pointer builder, Pointer model_data, String backend_name, String backend_config, PointerByReference dst_model_handle);

    void menoh_delete_model(Pointer model);

    int menoh_model_get_variable_dtype(Pointer model, String variable_name, IntByReference dst_dtype);

    int menoh_model_run(Pointer model);

    int menoh_model_get_variable_dims_size(Pointer model, String variable_name, IntByReference dst_size);

    int menoh_model_get_variable_dims_at(Pointer model, String variable_name, int index, IntByReference dst_size);

    int menoh_model_get_variable_buffer_handle(Pointer model, String variable_name, PointerByReference dst_data);
}
// CHECKSTYLE:ON
