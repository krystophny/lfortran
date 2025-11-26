subroutine slals0_mre(icompq, b, ldb, bx, ldbx, perm, k, work)
  implicit none
  integer :: icompq, ldb, ldbx, k
  integer :: perm(*)
  real :: b(ldb,*), bx(ldbx,*), work(*)
  real :: temp
  integer :: nrhs, info
  nrhs = 1

  info = 0
  if (icompq .eq. 0) then
     if (k .eq. 1) then
        call scopy(nrhs, bx, ldbx, b, ldb)
     else
        work(1) = -1.0
        temp = snrm2(k, work, 1)
        call sgemv('T', k, nrhs, 1.0, bx, ldbx, work, 1, 0.0, b(1,1), ldb)
        call slascl('G', 0, 0, temp, 1.0, 1, nrhs, b(1,1), ldb, info)
     end if
  else
     call sgemv('T', k, nrhs, 1.0, b, ldb, work, 1, 0.0, bx(1,1), ldbx)
     call scopy(nrhs, bx(1,1), ldbx, b(perm(1),1), ldb)
  end if
end subroutine

program lapack_04
  implicit none
  integer :: perm(2)
  real :: b(3,1), bx(3,1), work(2)
  perm = 1
  call slals0_mre(1, b, 3, bx, 3, perm, 2, work)
  print *, 'PASS'
end program
