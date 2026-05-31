program modules_77
    use modules_77_config, only: error_t, export_config, package_t
    implicit none

    type(package_t), target :: package
    type(package_t) :: copied
    type(error_t), allocatable :: error

    package%name = "hello"

    copied = export_config(package, error=error)

    if (allocated(error)) error stop 1
    if (copied%name /= "hello") error stop 2

    print *, "PASSED: modules_77"
end program modules_77
