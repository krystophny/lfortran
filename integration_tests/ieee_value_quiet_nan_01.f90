program ieee_value_quiet_nan_01
    use, intrinsic :: iso_fortran_env, only: dp => real64
    use, intrinsic :: ieee_arithmetic, only: ieee_is_nan, ieee_quiet_nan, ieee_value, &
        ieee_class_type
    implicit none

    real(dp) :: x, y
    type(ieee_class_type) :: cls

    x = 1.0d0
    cls = ieee_quiet_nan
    y = ieee_value(x, cls)
    if (.not. ieee_is_nan(y)) error stop 1
end program ieee_value_quiet_nan_01
