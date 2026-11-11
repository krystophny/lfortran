! An OpenMP THREADPRIVATE directive is rejected when the name is a named constant.
module openmp_threadprivate_08
    implicit none
    integer, parameter :: x = 1
    !$omp threadprivate(x)
end module
