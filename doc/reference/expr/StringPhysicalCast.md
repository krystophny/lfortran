# StringPhysicalCast

## ASR

<!-- BEGIN AUTO: asr -->
```
StringPhysicalCast(expr arg, string_physical_type old, string_physical_type new, ttype type, expr? value)
```
<!-- END AUTO: asr -->

## Documentation

_No documentation yet._

## Verify

<!-- BEGIN AUTO: verify -->
* x.`m_type` cannot be `nullptr`
* StringPhysicalCast should be of string type
* StringPhysicalCast argument must be a string
* StringPhysicalCast old physical type must match its argument
* StringPhysicalCast new physical type must match its result
* StringPhysicalCast return type shouldn't have length (Length should be implicit).
* StringPhysicalCast expression should have length kind of `ImplicitLength`.
<!-- END AUTO: verify -->
