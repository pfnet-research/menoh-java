package jp.preferred.menoh;

import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

public class DTypeTest {
    @Test
    public void floatDType() {
        assertAll("DType.FLOAT",
                () -> assertEquals(0, DType.FLOAT.getId()),
                () -> assertEquals(4, DType.FLOAT.size())
        );
    }

    @Test
    public void valueOfValidDType() {
        assertEquals(DType.FLOAT, DType.valueOf(0));
    }

    @Test
    public void valueOfIntMinDType() {
        MenohException e = assertThrows(MenohException.class, () -> DType.valueOf(Integer.MIN_VALUE));
        assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
    }

    @Test
    public void valueOfIntMaxDType() {
        MenohException e = assertThrows(MenohException.class, () -> DType.valueOf(Integer.MAX_VALUE));
        assertEquals(ErrorCode.UNDEFINED, e.getErrorCode());
    }
}
