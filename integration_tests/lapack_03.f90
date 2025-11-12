! Minimal reproducer for 0-based array with sequence association
! Tests A(0:*) passing A(1) to subroutine expecting 2D array
PROGRAM TEST
    IMPLICIT NONE
    REAL A(0:100)
    INTEGER M, N, LDA
    REAL ALPHA

    M = 6
    N = 4
    LDA = 7

    A = 1.0
    ALPHA = 2.0

    ! Pass A(1) from 0-based array to subroutine expecting 2D array
    CALL STRSM(3, N, ALPHA, A(1), LDA)

    PRINT *, 'Test completed successfully'
END PROGRAM

SUBROUTINE STRSM(M, N, ALPHA, A, LDA)
    IMPLICIT NONE
    INTEGER M, N, LDA
    REAL ALPHA
    REAL A(LDA, *)

    PRINT *, 'Accessing A(1,1) =', A(1,1)
    PRINT *, 'LDA =', LDA
    PRINT *, 'M =', M
    PRINT *, 'N =', N
END SUBROUTINE
