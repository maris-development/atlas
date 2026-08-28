# `AtlasWriter`

Builds one collection, then finishes. Nothing at the target is readable until
[`finish()`](#atlas.AtlasWriter.finish) writes the footer, and a collection
cannot be modified afterwards — see
[Immutability](../guides/immutability.md).

Use it as a context manager: a clean exit finishes the collection, an exception
abandons it.

::: atlas.AtlasWriter
    options:
        heading_level: 2
