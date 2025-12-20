program transfer_array_item_01
    use, intrinsic :: iso_fortran_env, only: int8, int64
    implicit none

    integer(int64) :: buf(12)
    integer(int64) :: remain
    integer(int8) :: buf8(8)

    buf = 0_int64
    buf(2) = 123456789_int64

    remain = 1_int64
    buf8 = 0_int8

    buf(remain + 1) = transfer(buf8, 0_int64)

    if (buf(2) /= 0_int64) error stop 1
end program transfer_array_item_01
