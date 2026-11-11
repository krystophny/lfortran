# CommonBlock

## ASR

<!-- BEGIN AUTO: asr -->
```
CommonBlock(symbol_table parent_symtab, identifier name, identifier? block_name, identifier link_name, symbol* members, common_association* associations, abi abi, storage_type storage, common_block_role role)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* `CommonBlock::m_name` cannot be `nullptr`
* ::`m_parent_symtab` cannot be `nullptr`
* is not the symbol of its own name in its parent symbol table
* is not in the symbol table it records as its parent
<!-- END AUTO: verify -->
