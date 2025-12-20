program intent_out_optional_alloc_twice_typebound
    use intent_out_optional_alloc_twice_typebound_m, only: container_t
    implicit none

    type(container_t) :: c
    integer, allocatable :: units(:)

    call c%configure(units)
    call c%configure(units)

    if (.not. allocated(units)) error stop 1
    if (size(units) /= 0) error stop 2
end program intent_out_optional_alloc_twice_typebound
