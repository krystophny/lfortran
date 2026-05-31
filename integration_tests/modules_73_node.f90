module modules_73_node
    use modules_73_base, only: value_t
    implicit none

    type :: node_t
        class(value_t), allocatable :: val
    end type node_t
end module modules_73_node
