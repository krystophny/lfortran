# ScopedVariableAttribute

## ASR

<!-- BEGIN AUTO: asr -->
```
ScopedVariableAttribute(symbol_table parent_symtab, identifier name, symbol variable, bool is_volatile, bool is_asynchronous)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* scoped attribute must belong to its declaring scope
* scoped attribute must specify volatile or asynchronous
* scoped attribute must reference a variable
* scoped attribute's variable must be visible in its scope
* scoped attribute's variable must be resolved
* scoped attribute must reference a variable entity
* local attributes must be recorded on the local variable
<!-- END AUTO: verify -->
