package jp.preferred.menoh;

public class MenohRunnerException extends RuntimeException {
    public MenohRunnerException(String message) {
        super(message);
    }

    public MenohRunnerException(String message, Throwable cause) {
        super(message, cause);
    }

    public MenohRunnerException(Throwable cause) {
        super(cause);
    }

    protected MenohRunnerException(
            String message,
            Throwable cause,
            boolean enableSuppression,
            boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }
}
