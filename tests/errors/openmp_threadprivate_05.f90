! An OpenMP THREADPRIVATE directive is rejected when a name in the list is empty.
module openmp_threadprivate_05
    implicit none
    integer :: x
    !$omp threadprivate(x,)
end module
