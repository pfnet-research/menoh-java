package jp.preferred.menoh;

public class VariableProfile {
    private final DType dtype;
    private final int[] dims;

    VariableProfile(DType dtype, int[] dims) {
        this.dtype = dtype;
        this.dims = dims;
    }

    DType dtype() {
        return this.dtype;
    }

    int[] dims() {
        return this.dims;
    }
}
