! An OpenMP THREADPRIVATE directive is rejected when the variable list is missing.
module openmp_threadprivate_01
    implicit none
    integer :: x
    !$omp threadprivate
end module
