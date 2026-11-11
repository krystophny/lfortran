# BlockData

## ASR

<!-- BEGIN AUTO: asr -->
```
BlockData(symbol_table symtab, identifier name, identifier? block_data_name, identifier* dependencies, location start_name, location end_name)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* The `BlockData::m_symtab` cannot be `nullptr`
* The `BlockData::m_symtab`->parent is not the right parent
* The `BlockData::m_symtab`'s parent must be TranslationUnit
* `BlockData::m_symtab`->counter must be unique
* The `X::m_symtab::asr_owner` must point to X
* BlockData name is required
<!-- END AUTO: verify -->
