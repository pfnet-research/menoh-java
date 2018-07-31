package jp.preferred.menoh;

/**
 * A representation of <code>menoh_error_code</code>.
 */
public enum ErrorCode {
    UNDEFINED(Integer.MIN_VALUE), // values[0]
    SUCCESS(0),
    STD_ERROR(1),
    UNKNOWN_ERROR(2),
    INVALID_FILENAME(3),
    UNSUPPORTED_ONNX_OPSET_VERSION(4),
    ONNX_PARSE_ERROR(5),
    INVALID_DTYPE(6),
    INVALID_ATTRIBUTE_TYPE(7),
    UNSUPPORTED_OPERATOR_ATTRIBUTE(8),
    DIMENSION_MISMATCH(9),
    VARIABLE_NOT_FOUND(10),
    INDEX_OUT_OF_RANGE(11),
    JSON_PARSE_ERROR(12),
    INVALID_BACKEND_NAME(13),
    UNSUPPORTED_OPERATOR(14),
    FAILED_TO_CONFIGURE_OPERATOR(15),
    BACKEND_ERROR(16),
    SAME_NAMED_VARIABLE_ALREADY_EXIST(17);

    private final int id;

    // cache values() to avoid cloning the backing array every time
    private static final ErrorCode[] values = values();

    ErrorCode(int id) {
        this.id = id;
    }

    public int getId() {
        return this.id;
    }

    /**
     * Returns the enum constant of the specified enum type with the specified ID.
     */
    public static ErrorCode valueOf(int value) throws MenohException {
        final int index = value + 1;
        if (1 <= index && index < values.length) {
            final ErrorCode ret = values[index];
            assert ret.id == value;

            return ret;
        } else {
            throw new MenohException(UNDEFINED, "The error code is undefined: " + value);
        }
    }
}
