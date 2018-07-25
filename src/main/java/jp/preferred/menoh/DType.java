package jp.preferred.menoh;

public enum DType {
    Float(0);

    private final int id;

    // cache values() to avoid cloning the backing array every time
    private static final DType[] values = values();

    DType(int id) {
        this.id = id;
    }

    public int getId() {
        return id;
    }

    public int size() throws MenohException {
        switch (this) {
            case Float:
                return 4;
            default:
                throw new MenohException(
                        ErrorCode.UNDEFINED, String.format("the size of dtype is unknown: %d", id));
        }
    }

    public static DType valueOf(int value) throws MenohException {
        if (0 <= value && value < values.length) {
            DType ret = values[value];
            assert ret.id == value;

            return ret;
        } else {
            throw new MenohException(ErrorCode.UNDEFINED, "undefined dtype: " + value);
        }
    }
}
