! An OpenMP THREADPRIVATE directive is rejected when the variable list is not closed.
module openmp_threadprivate_03
    implicit none
    integer :: x
    !$omp threadprivate(x
end module
