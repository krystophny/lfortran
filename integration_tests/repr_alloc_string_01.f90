program repr_alloc_string_01
    implicit none
    character(:), allocatable :: s
    character(:), allocatable :: r

    s = "abc"
    r = repr(s)

    if (index(r, " :: s =") == 0) error stop 1
    if (index(r, "= abc") == 0) error stop 2

    print *, r
end program repr_alloc_string_01
