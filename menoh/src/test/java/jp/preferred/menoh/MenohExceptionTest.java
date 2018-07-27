package jp.preferred.menoh;

import static jp.preferred.menoh.MenohException.checkError;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

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
                () -> assertEquals(" (std_error)", e.getMessage())
        );
    }

    @Test
    public void checkErrorUnknownErrorCode() {
        MenohException e = assertThrows(MenohException.class, () -> checkError(Integer.MAX_VALUE));
        assertAll("invalid ErrorCode",
                () -> assertEquals(ErrorCode.UNDEFINED, e.getErrorCode()),
                () -> assertEquals(" (2147483647)", e.getMessage())
        );
    }
}
