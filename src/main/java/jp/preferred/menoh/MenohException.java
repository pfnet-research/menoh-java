package jp.preferred.menoh;

import java.util.Locale;

public class MenohException extends RuntimeException {
    private final ErrorCode errorCode;

    public MenohException(ErrorCode errorCode, String message) {
        super(message);
        this.errorCode = errorCode;
    }

    public MenohException(ErrorCode errorCode, String message, Throwable cause) {
        super(message, cause);
        this.errorCode = errorCode;
    }

    public MenohException(ErrorCode errorCode, Throwable cause) {
        super(cause);
        this.errorCode = errorCode;
    }

    protected MenohException(
            ErrorCode errorCode,
            String message,
            Throwable cause,
            boolean enableSuppression,
            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
        this.errorCode = errorCode;
    }

    public ErrorCode getErrorCode() {
        return this.errorCode;
    }

    /**
     * Throws {@link MenohException} if the <code>errorCode</code> of a native Menoh function is
     * non-zero. This method must be called right after the invocation because it uses <code>
     * menoh_get_last_error_message</code>.
     *
     * @param errorCode an error code returned from the Menoh function
     * @throws MenohException
     */
    static void checkError(int errorCode) throws MenohException {
        if (errorCode != ErrorCode.SUCCESS.getId()) {
            final String errorMessage = MenohNative.INSTANCE.menoh_get_last_error_message();
            try {
                ErrorCode ec = ErrorCode.valueOf(errorCode);
                throw new MenohException(
                        ec,
                        String.format("%s: %s", ec.toString().toLowerCase(Locale.ENGLISH), errorMessage));
            } catch (MenohException e) {
                // jp.preferred.menoh.ErrorCode.valueOf() throws MenohException if the error code is
                // undefined
                throw new MenohException(ErrorCode.UNDEFINED, errorMessage, e);
            }
        }
    }
}
