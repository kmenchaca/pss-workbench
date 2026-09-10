# Retained experimental source

The public workbench lives under `pss/web/` and uses the forward-only engine in `pss/`. The root-level `argswarm`, `dreamlogic`, `embodied`, `evolution`, `knowledge`, `metalearner`, `negotiate`, `personas`, `redteam`, `temporal` and `unified` directories retain earlier experimental code and its tests.

These modules are not all exposed in the workbench. They are research prototypes, not promises of supported product features. Their source is retained so the engine's development remains inspectable. Historical result summaries, scoring claims and internal planning documents were omitted from this export because they were not independently reproduced for the public release; originals remain preserved in the source archive.

`demos`, `test_bugs`, `test_creative` and `test_prompts` contain experimental scenarios and deliberately buggy fixtures. They are not production code to copy into an application. The default workbench disables model-driven filesystem and command tools. Running older research entry points is a separate choice and can have different capabilities and provider costs.
