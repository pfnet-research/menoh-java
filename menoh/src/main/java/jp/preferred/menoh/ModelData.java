package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.PointerByReference;

import java.nio.ByteBuffer;

/**
 * <p>A model data.</p>
 *
 * <p>Make sure to {@link #close()} this object after finishing the process to free the underlying memory
 * in the native heap.</p>
 */
public class ModelData implements AutoCloseable {
    private Pointer handle;

    private DisposableMemory dataPtr;

    private ModelData(Pointer handle, DisposableMemory dataPtr) {
        this.handle = handle;
        this.dataPtr = dataPtr;
    }

    Pointer nativeHandle() {
        return this.handle;
    }

    DisposableMemory nativeDataPointer() {
        return this.dataPtr;
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
            if (dataPtr != Pointer.NULL) {
                dataPtr.dispose();
            }
        }
    }

    /**
     * <p>Loads an ONNX model from the specified byte array.</p>
     */
    public static ModelData fromOnnx(byte[] data) throws MenohException {
        return fromOnnx(data, 0, data.length);
    }

    /**
     * <p>Loads an ONNX model from the specified byte array.</p>
     */
    public static ModelData fromOnnx(byte[] data, int offset, int size) throws MenohException {
        final DisposableMemory dataPtr = BufferUtils.copyToNativeMemory(data, offset, size);
        final PointerByReference handle = new PointerByReference();
        checkError(
                MenohNative.INSTANCE.menoh_make_model_data_from_onnx_data_on_memory(
                        dataPtr.share(offset, size), size, handle));

        return new ModelData(handle.getValue(), dataPtr);
    }

    /**
     * <p>Loads an ONNX model from the specified byte buffer.</p>
     *
     * <p>If the specified <code>buffer</code> is direct, it will be directly used without copying.
     * Otherwise, it copies the content to a newly allocated buffer in the native heap ranging from
     * <code>position()</code> to <code>(limit() - 1)</code> without changing its position.</p>
     */
    public static ModelData fromOnnx(ByteBuffer data) throws MenohException {
        if (data == null || data.remaining() <= 0) {
            throw new IllegalArgumentException("data must not be null or empty");
        }

        final int size = data.remaining();
        final PointerByReference handle = new PointerByReference();

        if (data.isDirect()) {
            final Pointer dataPtr = BufferUtils.convertToNativeMemory(data);
            checkError(MenohNative.INSTANCE.menoh_make_model_data_from_onnx_data_on_memory(dataPtr, size, handle));

            return new ModelData(handle.getValue(), null); // doesn't free the memory
        } else {
            final DisposableMemory dataPtr = BufferUtils.copyToNativeMemory(data);
            checkError(MenohNative.INSTANCE.menoh_make_model_data_from_onnx_data_on_memory(dataPtr, size, handle));

            return new ModelData(handle.getValue(), dataPtr); // free the allocated memory after finished
        }
    }

    /**
     * <p>Loads an ONNX model from the specified file.</p>
     */
    public static ModelData fromOnnxFile(String onnxModelPath) throws MenohException {
        final PointerByReference handle = new PointerByReference();
        checkError(MenohNative.INSTANCE.menoh_make_model_data_from_onnx(onnxModelPath, handle));

        return new ModelData(handle.getValue(), null);
    }
}
