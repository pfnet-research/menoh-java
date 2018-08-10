package jp.preferred.menoh;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

public class ErrorCodeTest {
    @Test
    public void undefinedErrorCode() {
        assertEquals(Integer.MIN_VALUE, ErrorCode.UNDEFINED.getId());
    }

    @Test
    public void successErrorCode() {
        assertEquals(0, ErrorCode.SUCCESS.getId());
    }

    @Test
    public void valueOfValidErrorCode() {
        assertAll("valid ErrorCodes",
                () -> assertEquals(ErrorCode.SUCCESS, ErrorCode.valueOf(0)),
                () -> assertEquals(ErrorCode.UNSUPPORTED_INPUT_DIMS, ErrorCode.valueOf(18))
        );
    }

    @Test
    public void valueOfInvalidErrorCode() {
        assertAll("invalid ErrorCodes",
                () -> {
                    MenohException e = assertThrows(MenohException.class, () -> ErrorCode.valueOf(-1));
                    assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
                },
                () -> {
                    MenohException e = assertThrows(MenohException.class, () -> ErrorCode.valueOf(18 + 1));
                    assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
                });
    }

    @Test
    public void valueOfIntMinErrorCode() {
        MenohException e = assertThrows(MenohException.class, () -> ErrorCode.valueOf(Integer.MIN_VALUE));
        assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
    }

    @Test
    public void valueOfIntMaxErrorCode() {
        MenohException e = assertThrows(MenohException.class, () -> ErrorCode.valueOf(Integer.MAX_VALUE));
        assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
    }
}
