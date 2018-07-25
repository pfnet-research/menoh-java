package jp.preferred.menoh;

public class VariableProfile {
    private final DType dtype;
    private final int[] dims;

    VariableProfile(DType dtype, int[] dims) {
        this.dtype = dtype;
        this.dims = dims;
    }

    public DType dtype() {
        return this.dtype;
    }

    public int[] dims() {
        if (dims != null) {
            return this.dims.clone();
        } else {
            return new int[0];
        }
    }
}
