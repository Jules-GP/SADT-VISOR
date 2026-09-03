# CNE

Clinical Notes Extraction: read a folder of free-text clinical notes and write,
per note, the structured findings a fine-tuned language model extracts from it.

The model is a quantised GGUF run locally through llama.cpp. Nothing leaves the
machine, no request reaches an external API, and the weights are staged on the
server rather than downloaded at run time.

## What it does

For each note found under `notes`:

1. read it -- `.txt`, `.pdf` or `.docx`, whatever the encoding;
2. normalise the typography so the text looks the same to the model whichever
   format it came from;
3. **refuse it if there is no text**, rather than asking the model to extract
   findings from an empty string;
4. ask the model, once, with the system prompt its note type calls for;
5. write the answer as JSON, and the same answer rendered `key : value`.

`CNE_report.json` says what happened to every note.

## Arguments

| argument | what it is |
|---|---|
| `notes` | one note, or a folder of them. Searched recursively. |
| `notes_type` | `TMJ` or `Ortho`. Selects the system prompt and the context window together. |
| `model` | the hosted GGUF bundle, picked by name from `DATA/CNE/models/`. |
| `output_dir` | filled in by the server: the job's own output folder. |
| `max_tokens` | the longest answer per note (default 2048). |
| `temperature` | default `0.0`, so the same note extracts to the same answer twice. |
| `context_tokens` | `0` means the note type's default (TMJ 6144, Ortho 2048). |
| `seed` | the sampler's seed, for a reproducible run above temperature 0. |
| `device` | `cpu` (default) or `cuda`. See "The CPU build" below. |

`model` is a `Path` argument called `model`, which is what makes the server
publish it as a **name a client picks from what this server hosts** -- never a
file a clinician can upload, and never a download at run time.

## Provenance

Ported from DCBIA-OrthoLab/SlicerAutomatedDentalTools: `CNE_CLI/CNE_CLI.py`
(269 lines, the whole computation) and `CNE/CNE.py` (621 lines, the Qt panel).
The panel is not ported at all; what follows is about the CLI.

## Changes from upstream

Every one of these is a defect that lost or fabricated a clinical result, and
each has a test named after it in `tests/test_run.py`.

- **A scanned PDF is refused, not hallucinated from.** An image-only PDF has no
  text layer, so `extract_text` returned `""`. Upstream sent that to the model,
  which duly answered with a plausible extraction of a note it had never seen;
  it was written to disk and counted as a success. There is no `if not
  text.strip()` guard anywhere upstream. There is one here, the note is reported
  as failed by name, and nothing is written. CNE does not perform OCR and does
  not pretend to.
- **Three notes sharing a stem write three extractions.** The output name was
  `Extraction_<stem>.txt`, derived from the stem alone, while discovery accepted
  three extensions: `B_001.txt`, `B_001.pdf` and `B_001.docx` in one folder all
  wrote the same file with `open(..., "w")`. Two of the three were destroyed and
  the run reported 3/3. Output names keep the source extension
  (`Extraction_B_001.pdf.json`) and the output tree mirrors the input tree, so
  `a/note.txt` and `b/note.txt` cannot collide either. `plan_outputs` checks for
  a collision anyway before a single model call is made.
- **Discovery is recursive and case-insensitive.** `glob.glob(folder + "*.ext")`
  saw neither `NOTE.PDF` nor anything in a per-patient subfolder -- silently, with
  no message. A cohort is filed in subfolders; that is what a cohort is.
- **A missing reader costs the notes that need it.** Upstream called
  `sys.exit(1)` if `pymupdf` OR `python-docx` was missing, even for an
  all-`.txt` batch. Each reader is imported when a note of that format is
  actually read; a PDF that cannot be read is that note's failure, and a batch
  that was ALL PDFs answers 503 (`ToolUnavailableError`) rather than 422,
  because nothing the caller sends would make the package appear.
- **One note failing costs one note.** A blanket `except Exception: sys.exit(1)`
  wrapped the model load *and* the entire loop, so the first bad file ended the
  cohort.
- **An empty folder is a 422, not a finished run.** Upstream `sys.exit(0)`'d, so
  a mistyped path was reported to the user as a completed extraction.
- **The tally is what the run reports.** `successfully_processed` and
  `failed_files` were computed and then printed over by an unconditional "All
  files processed successfully!". 0 of 40 succeeding still exited 0. The summary
  counts what was *written*, and a run that wrote nothing raises.
- **The engine creates its own output directory.** Upstream's was created by the
  Slicer panel (`CNE/CNE.py:578`) and never by the CLI, so headless every write
  raised `FileNotFoundError`, was swallowed by the per-file `except`, and the run
  still claimed success.
- **`.docx` is read whole.** Upstream read `doc.paragraphs`, dropping every
  table, header and footer without a word. Clinical notes are frequently
  tabular. The body is walked in document order so a table stays attached to the
  sentence that introduces it, and a table row stays on one line -- split across
  lines, "Pain 7" reads as two unrelated notes.
- **`.txt` is decoded, not assumed.** Opened UTF-8 strict with no `errors=`, a
  note exported from Word on Windows raised inside the per-file `except`, was
  counted as a failure nobody saw, and the run still reported success.
  `utf-8-sig`, then `cp1252`, then `latin-1`, and the encoding that worked is in
  the report.
- **The note type cannot mean two things at once.** Upstream compared
  `notesType == "TMJ"` when choosing the context window (`:150`) and
  `notesType.upper() == "TMJ"` when choosing the system prompt (`:193`), so
  `"tmj"` got the TMJ instruction with the Ortho window of 2048 tokens and a
  long note was silently truncated. It is a `Literal["TMJ", "Ortho"]` here,
  validated by the server before `run()` is entered, and one table drives both.
- **Input and output folders are kept apart.** They could be the same, and the
  outputs matched the discovery glob, so a second run extracted from the first
  run's extractions. The same folder is refused; an output folder nested inside
  the notes folder is excluded from discovery.
- **A truncated answer is a failure.** `max_tokens=500` cut a long note's answer
  mid-JSON; the parse then failed and the truncated blob was written as if it
  were the result. `finish_reason` says exactly when that happened, the note is
  reported as failed, and nothing is written. The default is 2048 and it is an
  argument.
- **`temperature` defaults to 0.** Upstream used 0.1, so the same note extracted
  twice gave two different answers -- a clinical extraction that cannot be
  reproduced cannot be checked.
- **The JSON is kept.** Upstream flattened the model's object to `key : value`
  text at the very last step and threw the object away, so nothing downstream
  could read it. Both are written; the JSON is the canonical one (a value
  containing `" : "` is already ambiguous in the text form).
- **stderr is not surgically redirected.** Upstream `dup2`'d the process's
  stderr onto `/dev/null` around the model load and restored it *after* the load
  with no `try/finally` (`CNE/CNE.py:155-169`): a load that raised jumped the
  restore and left the whole interpreter writing into the void, traceback
  included. `quiet_stderr()` is a context manager, so the restore is on every
  path; what it captures is logged at DEBUG.
- **A model named for the other note type is warned about.** Pairing the Ortho
  weights with the TMJ instruction gives an answer that looks exactly right. It
  is a warning in the report and not a refusal, because a file name is not a
  contract and refusing on one would make a renamed bundle unusable.

## Deliberately not ported

- The `<filter-start>` / `<filter-progress>` / `<filter-comment>` stdout
  protocol: it exists to drive a `qSlicerCLIProgressBar`, and the HTTP contract
  here is one blocking request.
- Everything `slicer.*` and `qt.*`: the panel, the parameter node, the radio
  buttons, the CLI node observers, `qt.QStandardPaths` path construction.
- `check_lib_installed` / `install_function` / `check_dependencies`
  (`CNE/CNE.py:33-134`), which `pip_install` into the user's Slicer at the
  moment Apply is pressed. Dependencies are pinned in `pyproject.toml` and
  installed when the image is built.
- `getModelPath` (`:478-552`), which downloads a 4.4 GB GGUF from Hugging Face
  mid-run. A server holding patient data does not make outbound calls during a
  request; bundles are staged with `setup-models.sh --tool CNE`.
- `copyTestFiles` (`:409-476`), which `shutil.rmtree`s its destination
  unconditionally before copying.
- Writing the model's raw answer when it holds no JSON. That was upstream's
  fallback and it is exactly what makes a bad answer indistinguishable from a
  good one in the output folder. The note is reported as failed with the answer's
  length; the answer itself is never logged, being derived from patient data.

## The CPU build

`llama-cpp-python` is pinned to the CPU wheel index
(`https://abetlen.github.io/llama-cpp-python/whl/cpu`), which is what upstream
installs too (`CNE/CNE.py:62`). These models are 4-bit quantisations that run on
a CPU, and the CUDA build would put a second CUDA runtime in an image whose
other tools already carry one.

`device` still offers `cuda`, and it maps to `n_gpu_layers=-1`. On the CPU build
that offloads nothing and costs nothing; swapping the index above for
`.../whl/cu124` is the entire change a GPU deployment needs. Declaring the
argument at all is also what lets the server see that a CNE run is CPU work, so
it does not queue behind a segmentation for the card.

## Data

`DATA/CNE/models/<bundle>/` -- one folder per fine-tune, each holding one
`.gguf`. `scripts/data-manifest.yml` in `VISOR-serve` stages the TMJ weights
(`TMJ/qwen-ft-q4_k_m.gguf`, 4.4 GB) from
`dcbia/Qwen-2.5-7B-Instruct-TMJ`. The Ortho bundle
(`dcbia/Meta-Llama-3.1-8B-Instruct-Ortho`) answers 404 today -- the repository is
private, renamed or gone -- so it is not listed; `notes_type = "Ortho"` works as
soon as a bundle for it is staged.

A bundle folder holding exactly one `.gguf` resolves to it, recursively. One
holding several is **refused by name**: which model vintage ran must never
depend on sort order.

## Working on it

```bash
cd tools/CNE
uv sync --all-groups
uv run pytest                                   # 122 tests, no weights, no GPU
uv run pytest -m models -o addopts=             # needs CNE_MODEL=<bundle>
.venv/bin/python -m pyflakes src tests
.venv/bin/python ../../scripts/describe.py .    # the schema the server publishes
```
