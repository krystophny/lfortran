C Test FORTRAN 77 sequence association with assumed-size arrays
C This pattern is common in LAPACK where array elements are passed
C to subroutines expecting array pointers
      SUBROUTINE CALLER(A, N)
      IMPLICIT NONE
      INTEGER N
      REAL A(N, *)
      EXTERNAL SUB
C Pass array element as start of array - FORTRAN 77 sequence association
      CALL SUB(A(1,1), N)
      END

      SUBROUTINE SUB(X, N)
      IMPLICIT NONE
      INTEGER N
      REAL X(*)
      X(1) = 42.0
      END

      PROGRAM TEST
      REAL A(10, 10)
      CALL CALLER(A, 10)
      IF (A(1,1) /= 42.0) ERROR STOP
      PRINT *, 'PASS'
      END
