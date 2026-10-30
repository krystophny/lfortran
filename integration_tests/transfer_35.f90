program transfer_35
  ! TRANSFER with an array-section mold: the result holds the smallest number
  ! of mold elements that cover the source, so its size follows the byte sizes
  ! of source and mold element and not the bounds of the section
  ! (F2018 16.9.193).
  implicit none
  integer(1) :: b(13)
  integer(8) :: q
  integer :: w

  w = 258
  q = 258_8
  if (size(transfer(w, b(1:4))) /= 4) error stop "size of transfer(w, b(1:4))"
  if (size(transfer(w, b(3:6))) /= 4) error stop "size of transfer(w, b(3:6))"
  if (size(transfer(w, b(5:8))) /= 4) error stop "size of transfer(w, b(5:8))"
  if (size(transfer(w, b(1:13:3))) /= 4) error stop "size with a strided section"
  if (size(transfer(q, b(5:8))) /= 8) error stop "size of transfer(q, b(5:8))"
  if (lbound(transfer(w, b(5:8)), 1) /= 1) error stop "lower bound of the result"

  b = 0
  b(5:8) = transfer(w, b(5:8))
  if (any(b(5:8) /= [2_1, 1_1, 0_1, 0_1])) error stop "wrong bytes in b(5:8)"
  if (any(b(1:4) /= 0) .or. any(b(9:13) /= 0)) error stop "write outside b(5:8)"

  print *, "test passed"
end program
