! Test for intent(out) nested allocatable deallocation
! Issue #9097: nested allocatables should be deallocated at procedure entry
!
! Fixed by PR #9039: intent(out) now properly deallocates nested allocatables
!
! Verification: Run with valgrind to confirm no memory leaks
!   gfortran:   All heap blocks freed (correct)
!   ifx:        All heap blocks freed (correct)
!   nvfortran:  No leaks (correct)
!   lfortran:   All heap blocks freed (correct - after PR #9039)
program intent_out_nested_dealloc
    implicit none

    type :: node_t
        integer, allocatable :: data(:)
    end type

    type(node_t), allocatable :: nodes(:)

    ! First call - allocates nodes and nested data
    call create_nodes(nodes, 3, 5)

    ! Second call - should deallocate nested data(:) before reallocating
    call create_nodes(nodes, 2, 4)

    if (size(nodes) /= 2) error stop "Expected 2 nodes"
    if (size(nodes(1)%data) /= 4) error stop "Expected data size 4"

    ! Explicit cleanup to isolate intent(out) leak from scope-exit leak
    call cleanup_nodes(nodes)

    print *, "Test passed"

contains

    subroutine cleanup_nodes(nodes)
        type(node_t), allocatable, intent(inout) :: nodes(:)
        integer :: i
        if (allocated(nodes)) then
            do i = 1, size(nodes)
                if (allocated(nodes(i)%data)) deallocate(nodes(i)%data)
            end do
            deallocate(nodes)
        end if
    end subroutine

    subroutine create_nodes(nodes, n, sz)
        type(node_t), allocatable, intent(out) :: nodes(:)
        integer, intent(in) :: n, sz
        integer :: i

        allocate(nodes(n))
        do i = 1, n
            allocate(nodes(i)%data(sz))
            nodes(i)%data = i * 10
        end do
    end subroutine

end program
