! An OpenMP THREADPRIVATE directive is rejected when text follows the variable list.
module openmp_threadprivate_04
    implicit none
    integer :: x
    !$omp threadprivate(x) extra
end module
