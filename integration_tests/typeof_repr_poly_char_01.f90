program typeof_repr_poly_char_01
    implicit none
    class(*), allocatable :: u
    character(:), allocatable :: s

    allocate(character(3) :: u)
    select type(u)
    type is (character(*))
        u = "xyz"
    end select

    s = typeof(u)
    if (index(s, "character") /= 1) error stop 1

    s = repr(u)
    if (index(s, "character :: u =") /= 1) error stop 2
    if (index(s, "= xyz") == 0) error stop 3

    print *, typeof(u)
    print *, repr(u)
end program typeof_repr_poly_char_01
