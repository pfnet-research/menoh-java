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
    ONNX_PARSE_ERROR(4),
    INVALID_DTYPE(5),
    INVALID_ATTRIBUTE_TYPE(6),
    UNSUPPORTED_OPERATOR_ATTRIBUTE(7),
    DIMENSION_MISMATCH(8),
    VARIABLE_NOT_FOUND(9),
    INDEX_OUT_OF_RANGE(10),
    JSON_PARSE_ERROR(11),
    INVALID_BACKEND_NAME(12),
    UNSUPPORTED_OPERATOR(13),
    FAILED_TO_CONFIGURE_OPERATOR(14),
    ERROR_CODE(15),
    SAME_NAMED_VARIABLE_ALREADY_EXIST(16);

    private final int id;

    // cache values() to avoid cloning the backing array every time
    private static final ErrorCode[] values = values();

    ErrorCode(int id) {
        this.id = id;
    }

    public int getId() {
        return this.id;
    }

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
