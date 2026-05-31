module modules_77_config
    implicit none
    private

    public :: error_t, export_config, package_t, string_t

    type :: string_t
        character(len=:), allocatable :: s
    end type string_t

    type :: error_t
        character(len=:), allocatable :: message
    end type error_t

    type :: package_t
        character(len=:), allocatable :: name
        type(string_t), allocatable :: profiles(:)
    end type package_t

contains

    function export_config(self, features, profile, verbose, error) result(cfg)
        class(package_t), intent(in), target :: self
        type(string_t), optional, intent(in), target :: features(:)
        character(len=*), optional, intent(in) :: profile
        logical, optional, intent(in) :: verbose
        type(error_t), allocatable, intent(out) :: error
        type(package_t) :: cfg

        type(string_t), pointer :: want_features(:)
        logical :: apply_default

        if (present(profile) .and. present(features)) then
            allocate(error)
            error%message = "conflicting options"
            return
        end if

        cfg = self

        if (present(features)) then
            want_features => features
        elseif (present(profile)) then
            apply_default = profile == "debug"
            want_features => self%profiles
        else
            nullify(want_features)
            apply_default = .true.
        end if

        if (apply_default .and. associated(want_features)) then
            if (present(verbose)) then
                if (verbose) error stop 1
            end if
        end if
    end function export_config

end module modules_77_config
