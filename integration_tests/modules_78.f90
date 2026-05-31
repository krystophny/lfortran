module modules_78_alloc_sret_final
    implicit none

    type :: request_t
        logical :: on = .false.
        character(len=:), allocatable :: name
        character(len=:), allocatable :: version
    end type request_t

    type :: config_t
        type(request_t) :: openmp
        type(request_t) :: stdlib
    contains
        final :: config_final
        procedure :: reset => config_reset
    end type config_t

    type :: package_t
        character(len=:), allocatable :: name
        type(config_t) :: meta
    contains
        procedure :: export_config
    end type package_t

contains

    subroutine request_destroy(self)
        class(request_t), intent(inout) :: self

        self%on = .false.
        if (allocated(self%version)) deallocate(self%version)
        if (allocated(self%name)) deallocate(self%name)
    end subroutine request_destroy

    subroutine config_final(self)
        type(config_t), intent(inout) :: self

        call self%reset()
    end subroutine config_final

    subroutine config_reset(self)
        class(config_t), intent(inout) :: self

        call request_destroy(self%openmp)
        self%openmp%name = "openmp"
        call request_destroy(self%stdlib)
        self%stdlib%name = "stdlib"
    end subroutine config_reset

    function export_config(self) result(cfg)
        class(package_t), intent(in), target :: self
        type(package_t) :: cfg

        cfg = self
    end function export_config

    subroutine run_alloc_sret()
        type(package_t), target :: source
        type(package_t), allocatable :: copied

        call source%meta%reset()
        source%name = "hello"
        allocate(copied)
        copied = source%export_config()
        call copied%meta%reset()
        if (source%meta%openmp%name /= "openmp") error stop 1
    end subroutine run_alloc_sret

end module modules_78_alloc_sret_final

program modules_78
    use modules_78_alloc_sret_final, only: run_alloc_sret
    implicit none

    call run_alloc_sret()
    print *, "PASSED: modules_78"
end program modules_78
