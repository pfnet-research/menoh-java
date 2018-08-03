package jp.preferred.menoh;

/**
 * A variable profile added to {@link VariableProfileTable}.
 */
public class VariableProfile {
    private final DType dtype;
    private final int[] dims;

    VariableProfile(DType dtype, int[] dims) {
        this.dtype = dtype;
        this.dims = dims;
    }

    /**
     * A data type of the variable.
     */
    public DType dtype() {
        return this.dtype;
    }

    /**
     * An array of dimensions.
     */
    public int[] dims() {
        if (dims != null) {
            return this.dims.clone();
        } else {
            return new int[0];
        }
    }
}
