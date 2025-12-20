program formatted_read_complex_array_01
    use, intrinsic :: iso_fortran_env, only: sp => real32
    implicit none

    integer :: unit
    complex(sp) :: z(2)

    unit = 10
    open(unit=unit, status="scratch", action="readwrite", form="formatted")
    write(unit, "(a)") "(1.0,2.0) (3.0,4.0)"
    rewind(unit)

    read(unit, "(2f20.0)") z
    close(unit)

    if (abs(real(z(1), kind=sp) - 1.0_sp) > 1.0e-6_sp) error stop 1
    if (abs(aimag(z(1)) - 2.0_sp) > 1.0e-6_sp) error stop 2
    if (abs(real(z(2), kind=sp) - 3.0_sp) > 1.0e-6_sp) error stop 3
    if (abs(aimag(z(2)) - 4.0_sp) > 1.0e-6_sp) error stop 4
end program formatted_read_complex_array_01
