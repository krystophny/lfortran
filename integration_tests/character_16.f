! Test: CHARACTER(*) assumed-size array passed through EXTERNAL declaration
! This tests that implicit interface correctly uses Source ABI (not BindC)
! for Fortran CHARACTER arrays with EXTERNAL procedures
      PROGRAM TEST
      CHARACTER ADUMMA(1)
      ADUMMA(1) = ' '
      CALL SCHKHS(ADUMMA)
      END

      SUBROUTINE SCHKHS(ADUMMA)
      CHARACTER ADUMMA(1)
      EXTERNAL SLATME
      CALL SLATME(ADUMMA)
      END

      SUBROUTINE SLATME(EI)
      CHARACTER EI(*)
      IF (EI(1) == ' ') THEN
         PRINT *, 'PASS'
      ELSE
         ERROR STOP 'FAIL: expected space character'
      END IF
      RETURN
      END
