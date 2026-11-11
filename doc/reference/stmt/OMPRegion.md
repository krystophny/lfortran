# OMPRegion

## ASR

<!-- BEGIN AUTO: asr -->
```
OMPRegion(omp_region_type region, omp_clause* clauses, stmt* body, exec_target exec_target)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* OMPOverwrite may only be a clause of a parallel loop assigned to the device
* the array of an OMPOverwrite clause must be an array variable or a component of one
* the condition of an OMPOverwrite clause must be a scalar logical
* a variable named by an OMPRegion clause must be a Var
<!-- END AUTO: verify -->
