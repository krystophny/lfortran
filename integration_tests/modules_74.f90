module modules_74_feature
    implicit none

    type :: request_t
        logical :: on = .false.
        character(len=:), allocatable :: name
        character(len=:), allocatable :: version
    end type request_t

    type :: meta_t
        type(request_t) :: mpi
        type(request_t) :: openmp
    contains
        procedure :: reset => meta_reset
        final :: meta_final
    end type meta_t

    type :: feature_t
        type(meta_t) :: meta
        integer :: id = 0
    end type feature_t

    type :: collection_t
        type(feature_t), allocatable :: variants(:)
    contains
        procedure :: push => push_variant
    end type collection_t

contains

    elemental subroutine request_destroy(self)
        type(request_t), intent(inout) :: self

        self%on = .false.
        if (allocated(self%version)) deallocate(self%version)
        if (allocated(self%name)) deallocate(self%name)
    end subroutine request_destroy

    elemental subroutine meta_reset(self)
        class(meta_t), intent(inout) :: self

        call request_destroy(self%mpi)
        call request_destroy(self%openmp)
        self%mpi%name = "mpi"
        self%openmp%name = "openmp"
    end subroutine meta_reset

    subroutine meta_final(self)
        type(meta_t), intent(inout) :: self

        call self%reset()
    end subroutine meta_final

    function make_feature(id) result(feature)
        integer, intent(in) :: id
        type(feature_t) :: feature

        feature%id = id
        feature%meta%mpi%on = .true.
        feature%meta%mpi%name = "mpi"
        feature%meta%mpi%version = "*"
        feature%meta%openmp%name = "openmp"
    end function make_feature

    elemental subroutine push_variant(self, variant)
        class(collection_t), intent(inout) :: self
        type(feature_t), intent(in) :: variant

        type(feature_t), allocatable :: tmp(:)
        integer :: n

        if (.not. allocated(self%variants)) then
            allocate(self%variants(1), source=variant)
        else
            n = size(self%variants)
            allocate(tmp(n + 1))
            if (n > 0) tmp(1:n) = self%variants
            tmp(n + 1) = variant
            call move_alloc(tmp, self%variants)
        end if
    end subroutine push_variant

    function default_feature() result(collection)
        type(collection_t) :: collection

        call collection%push(make_feature(1))
        call collection%push(make_feature(2))
    end function default_feature

end module modules_74_feature

program modules_74
    use modules_74_feature, only: collection_t, default_feature
    implicit none

    type(collection_t), allocatable :: collections(:)

    allocate(collections(2))
    collections(1) = default_feature()
    collections(2) = default_feature()

    if (size(collections(1)%variants) /= 2) error stop 1
    if (collections(1)%variants(1)%meta%mpi%name /= "mpi") error stop 2
    if (collections(1)%variants(2)%meta%mpi%version /= "*") error stop 3
    if (collections(2)%variants(2)%id /= 2) error stop 4

    print *, "PASSED: modules_74"
end program modules_74
