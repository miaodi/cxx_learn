# Reference Implementation

This tree contains complete headers for checking the course specification. It
is deliberately excluded from the default include path.

Work through `../include/` and the tests first. When you need to compare an API
or verify the full course, configure a separate build with:

```sh
-DEXPR_AD_MILESTONE=6 -DEXPR_AD_USE_REFERENCE=ON
```

Avoid copying a whole reference header into the learner tree. Compare one rule
at a time after you can explain why your current implementation fails.
