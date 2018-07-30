package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;

// CHECKSTYLE:OFF
import static org.junit.jupiter.api.Assertions.*;
// CHECKSTYLE:ON

import org.junit.jupiter.api.Test;

public class MenohExceptionTest {
    @Test
    public void checkErrorSuccess() {
        checkError(ErrorCode.SUCCESS.getId());
    }

    @Test
    public void checkErrorFailed() {
        MenohException e = assertThrows(MenohException.class, () -> checkError(ErrorCode.STD_ERROR.getId()));
        assertAll("ErrorCode.STD_ERROR",
                () -> assertEquals(ErrorCode.STD_ERROR, e.getErrorCode()),
                () -> assertTrue(e.getMessage().endsWith(" (std_error)"),
                        String.format("%s doesn't end with \"(std_error)\".", e.getMessage()))
        );
    }

    @Test
    public void checkErrorUnknownErrorCode() {
        MenohException e = assertThrows(MenohException.class, () -> checkError(Integer.MAX_VALUE));
        assertAll("invalid ErrorCode",
                () -> assertEquals(ErrorCode.UNDEFINED, e.getErrorCode()),
                () -> assertTrue(e.getMessage().endsWith(" (2147483647)"),
                        String.format("%s doesn't end with \"(2147483647)\".", e.getMessage()))
        );
    }
}
