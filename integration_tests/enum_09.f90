module enum_09_m
    implicit none

    enum, bind(c)
        enumerator :: id_all = -1, id_unknown, id_gcc
    end enum

    integer, parameter :: compiler_enum = kind(id_unknown)
    integer, parameter :: os_all = -1

    type :: platform_config_t
        integer(compiler_enum) :: compiler = id_all
        integer :: os_type = os_all
    end type

    interface platform_config_t
        module procedure new_platform_id
    end interface

contains

    type(platform_config_t) function new_platform_id(compiler_id, os_type)
        integer(compiler_enum), intent(in) :: compiler_id
        integer, intent(in) :: os_type

        new_platform_id%compiler = compiler_id
        new_platform_id%os_type = os_type
    end function

end module

program enum_09
    use enum_09_m, only: id_all, id_unknown, os_all, platform_config_t
    implicit none

    type(platform_config_t) :: platform

    platform = platform_config_t(id_all, os_all)

    if (id_all /= -1) error stop
    if (id_unknown /= 0) error stop
    if (platform%compiler /= id_all) error stop
    if (platform%os_type /= os_all) error stop
end program
