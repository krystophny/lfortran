program allocatable_polymorphic_mold_02
    use allocatable_polymorphic_mold_02_base, only: error_t
    use allocatable_polymorphic_mold_02_child, only: child_t, child_revision
    implicit none

    type(child_t) :: obj
    type(error_t), allocatable :: error

    obj = child_revision("https://example.invalid/project.git", &
                         "7264878cdb1baff7323cc48596d829ccfe7751b8")
    call obj%roundtrip(error)
    if (allocated(error)) error stop
end program allocatable_polymorphic_mold_02
