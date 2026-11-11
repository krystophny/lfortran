# StringFormat

## ASR

<!-- BEGIN AUTO: asr -->
```
StringFormat(expr? fmt, expr* args, string_format_kind kind, dtio_binding* dtio_bindings, expr? dtio_unit, ttype type, expr? value)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* `StringFormat::m_fmt` must be nil for FormatFortranLeadingBlank
* `StringFormat::m_dtio_bindings` must be empty unless the format kind is FormatFortran or FormatFortranLeadingBlank
* `StringFormat::m_dtio_unit` must be nil when `m_dtio_bindings` is empty
* `StringFormat::m_dtio_unit` must be an integer scalar
* `StringFormat::m_dtio_unit` must be a Var or an IntegerConstant
<!-- END AUTO: verify -->
