! Minimal reproducer for BLAS snrm2.f90 tokenizer error
! Error: "Expecting terminating symbol for function" at line with result type declaration
! From: LAPACK build with --std=legacy

function SNRM2(n, x, incx)
   integer, parameter :: wp = kind(1.e0)
   real(wp) :: SNRM2

   integer, intent(in) :: n, incx
   real(wp), intent(in) :: x(*)

   SNRM2 = 0.0_wp
end function

program test_snrm2
   integer, parameter :: wp = kind(1.e0)
   real(wp) :: SNRM2
   real(wp) :: x(3)

   x = [1.0_wp, 2.0_wp, 3.0_wp]
   print *, SNRM2(3, x, 1)
end program
