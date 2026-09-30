# Portable Models And Results

Use a separate folder for the files you intend to share. Keep original runs and
historical model folders unchanged. The public bundle helpers create and check
ordinary ZIP files; they do not publish files or start experiments.

The [experiment scripts](../../scripts/paper_experiments/README.md) show three
complete routes: rebuild reports, evaluate saved actors, and repeat a declared
study. Each route needs its own source/environment and input records. A small
software check does not reproduce a paper's scientific findings.

## Prepare A Bundle

First collect models with `training.collect_actors` or export a chosen model with
`training.export_actor`. A collection carries `catalog.json`, model folders and
checked original descriptions under `checkpoints/`. Those descriptions preserve
training-history evidence after moving the library. They are not full learner
states and cannot resume training. Keep the whole collection together.

For saved results, copy the whole result folder, including its descriptions,
summaries, game rows and any selected replays. Keep associated selection decisions
and the raw evidence they refer to when preparing a complete study release.
The bundle tool preserves files exactly; it does not repair broken paths inside
a partial study or rewrite a historical identity.

Include applicable license notices and a source/environment record. Then:

```python
import json
from pathlib import Path

import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

# These are the actual compatible reader/source facts prepared for this release.
# Each model also keeps its own original training-source record.
facts = json.loads(Path("/data/release-facts.json").read_text())
record = marl_bgs.export_bundle(
    "/data/prepared-model-library",
    output_dir="/data/model-download",
    metadata=facts,
)
print(record)
```

`release-facts.json` has exactly four fields: `source` and `environment` are
nonempty evidence objects; `license` and `citation` are nonempty text. Fill them
with the actual records and applicable terms. Do not invent a source revision,
license or citation. The tool preserves these facts; it does not certify them.

The exact new destination contains:

```text
model-download/
  bundle.zip
  download.json
```

The JSON record lists every relative file, its byte count and SHA-256, plus the
archive's size and hash. SHA-256 is a fingerprint of file bytes. It detects
changes; it does not prove the model is good or that an experiment was fair.
An occupied export destination is refused. Symbolic links, special files, Git
history, bytecode caches and private milestone notes are refused on both routes.

The exporter streams files through a small buffer. During creation, allow disk
space for the originals plus the new compressed archive. Compression and copying
still take time; report measured time and bytes for the actual released bundle.
Do not use a tiny fixture's timing as a large-library estimate.

## Open A Local Bundle Or Download Explicitly

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

library = marl_bgs.prepare_bundle(
    "/data/model-download/download.json",
    output_dir="/data/ready-models",
)
print(library)
```

Move `download.json` and its archive together. Their paths are relative to the
record. Extraction checks archive and member hashes before publishing the new
folder. A complete identical destination is checked and reused. A conflicting
folder is left untouched. Missing, extra, duplicate, linked or damaged members
are refused. Interrupted unpublished extraction is cleaned up.

A publisher can later add `archive.url` to the download record. The URL must use
HTTP or HTTPS and serve the exact recorded bytes. Download is explicit:

```python
library = marl_bgs.prepare_bundle(
    "/data/download.json", output_dir="/data/ready-models", download=True
)
```

A missing archive may then be downloaded once. A corrupt local archive raises an
error; it is not silently replaced. There is no automatic download at import,
help, notebook opening or training start. Extraction needs temporary space for
the unpacked contents and, for a download, the compressed archive. A released
record must state actual sizes, hosting and compatible source/environment.
These local tools do not imply that any public model download currently exists.

Use the existing loader afterward. No model is loaded during bundle preparation:

```python
from marl_battlegrounds import training

system = training.load_system(library / "ACTUAL-RELATIVE-ACTOR-PATH-FROM-CATALOG")
```

Use the actor path from `catalog.json`; do not guess a name or select a model just
because it is the last file. Catalog roles such as Selected retain their saved
selection evidence. They are not a claim of universal best performance.

## Prepare An Anonymous Actor Copy

Anonymous preparation is separate from publication. The supported call is:

```python
import marl_battlegrounds as marl_bgs
from marl_battlegrounds import training

copy = training.export_actor(
    "/data/original-actor", output_dir="/data/review-model", anonymous=True
)
print(copy)
```

This creates a separate derivative. It keeps actor payload bytes, inference
identity, input scale, spawn frame, parameter sharing, counters and required PQN
running statistics. The description gets a new export ID, `derived_from` naming
the input artifact, and an `anonymous_copy` text-scan record. Original files are
never rewritten. Exporting an already checked anonymous copy is idempotent.

A capture keeps its complete capture identity and content binding. Human-readable
run labels receive stable digest aliases. Only named content audit paths may be
projected; the existing schema checks that scientific content is unchanged.
A checkpoint actor keeps its original opaque run ID and an inventoried original
learner description in `origin/<checkpoint-id>/checkpoint_details.json`. Its
normal actor loader works after relocation. The original description supplies
provenance only; it is not a recoverable training state.

The exporter preserves known, none and unknown opponent exposure, controller
identities and inherited evidence. It never turns missing or unknown history
into verified none. A full learner becomes a verified actor export; its artifact
kind changes, while the evidence remains checked. Historical schema-1 actors
may only be copied unchanged, with existing self-contained provenance. Anonymous
extraction of a historical learner is refused.

**This is a narrow preparation route.** It refuses unknown metadata fields,
identifying text, unsafe required paths and unclassified custom records before
creating the destination. In particular, ordinary later trainer checkpoints
often carry private paths in host or recovery records. They may be refused.
Required original descriptions are not edited to make the copy pass. A refusal
names the blocking field; do not drop that evidence or silently omit a model
from a promised complete bundle. Custom methods still work in normal exports;
anonymous admission does not redefine the `System` API.

The scan checks classified metadata and supported storage metadata for local
user/host names, locations, URLs and email addresses. Opaque array storage stays
unchanged. This is not a guarantee against every possible identity inference.
Model hashes can link to outside records. Review the complete distribution,
including catalogs, configs, source files and license notices, before submission.
Keep the original-to-copy mapping in a separate private folder outside the ZIP.
Public configs/catalogs must reference derivative identities and relative paths.

The generic `export_bundle` function does not anonymize its input. Assemble the
checked review copies and required notices first, then bundle them. Preserve
required attribution. No named account link, source history or private log should
enter an anonymous submission. Publication and the venue's actual submission
requirements still need their separate review; these tools do not claim venue
compliance or upload anything.
