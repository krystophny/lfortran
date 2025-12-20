program transfer_11
  use, intrinsic :: iso_fortran_env, only: int8, int16
  implicit none

  integer(int16) :: key(2), key2(2)
  integer(int8) :: bytes(2*size(key))

  key = [int(1234, int16), int(-7, int16)]

  bytes = transfer(key, 0_int8, 2*size(key))
  key2 = transfer(bytes, key2)

  if (any(key2 /= key)) error stop
  print *, "ok"
end program transfer_11
