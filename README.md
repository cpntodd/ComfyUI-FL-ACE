# Cpntodd's ComfyUI AceStep Lab

This is **Cpntodd's personal ComfyUI fork for local music AI work**.

The project is centred on [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5): preparing personal music datasets, generating useful musical metadata, training LoRAs, and testing the results locally. ComfyUI provides the visual workflow, model management, device handling, and a place to iterate on the whole process.

This repository is not intended to be a general-purpose mirror of every ComfyUI feature. It is a working local AI laboratory for exploring how far a privately controlled AceStep pipeline can go on Cpntodd's hardware and data.

## What this fork is for

- Build and test AceStep music-generation workflows locally.
- Prepare songs for training from audio, metadata, and lyric files.
- Train personal LoRAs for styles, voices, genres, and musical ideas.
- Compare CPU language-model labelling with GPU/XPU training and generation.
- Improve dataset quality through repeatable, inspectable node workflows.
- Experiment with local AI tooling without sending the music collection or lyrics to a hosted service.

The current hardware target is a Linux workstation with an Intel Arc B580 GPU. The code is being developed around real local constraints: shared system memory, GPU memory pressure, CPU language-model throughput, and the need to keep the training workflow usable while other AI components are active.

## AceStep workflow

The main workflow is:

```text
audio + lyrics
    -> scan dataset
    -> build multilingual lyric context
    -> CPU LLM auto-labelling
    -> VAE/CLIP preprocessing
    -> LoRA training
    -> local AceStep generation and evaluation
```

Song folders can contain Macedonian, transliterated, and English lyric Markdown files. The labelling path uses those files as verified context so the CPU language model can preserve language, meaning, cultural references, and song-specific terminology instead of guessing from audio alone. This is especially important for Macedonian material, regional instruments, and terms that a general-purpose captioning model may not recognise reliably.

## Included AceStep nodes

The custom node package is in [`custom_nodes/ComfyUI-FL-AceStep-Training`](custom_nodes/ComfyUI-FL-AceStep-Training).

| Node | Purpose |
| --- | --- |
| `FL AceStep LLM Loader` | Loads the 5Hz AceStep language model for audio understanding and labelling. |
| `FL AceStep Scan Audio Directory` | Finds audio and reads sidecar metadata plus folder-level lyric variants. |
| `FL AceStep Auto-Label Samples` | Produces captions, BPM, key, genre, lyrics, and related training metadata. |
| `FL AceStep Preprocess Dataset` | Encodes audio and text into tensors for training. |
| `FL AceStep Training Configuration` | Defines LoRA and training settings. |
| `FL AceStep Train LoRA` | Runs the AceStep LoRA training loop and reports progress in ComfyUI. |

The auto-labelling node supports bounded batches for the CPU LLM. This keeps the workflow ordered while allowing more than one sample to be processed at a time. CPU generation threads are expanded temporarily for the labelling operation and restored afterward so the rest of ComfyUI retains normal resource behaviour.

## Quick start

1. Start ComfyUI using your local launcher and confirm that the AceStep custom nodes load without errors.
2. Load the AceStep checkpoint, VAE, and text encoder with ComfyUI's native loaders.
3. Load an AceStep 5Hz LLM if automatic labelling is required.
4. Scan the song directory with `FL AceStep Scan Audio Directory`.
5. Run `FL AceStep Auto-Label Samples`, supplying the AceStep model and VAE.
6. Review the generated metadata, then preprocess the dataset.
7. Configure and run LoRA training.
8. Save the trained LoRA and test it in a separate AceStep generation workflow.

For a starting point, see [`workflow/B580-XPU-Training.json`](custom_nodes/ComfyUI-FL-AceStep-Training/workflow/B580-XPU-Training.json) and [`workflow/Example-WF.json`](custom_nodes/ComfyUI-FL-AceStep-Training/workflow/Example-WF.json).

## Dataset conventions

The scanner supports common audio formats including `.wav`, `.mp3`, `.flac`, `.ogg`, `.opus`, and `.m4a`.

A song folder may include:

```text
song/
├── song.wav
├── song.txt             # optional legacy lyric sidecar
├── mk_lyrics.md         # Macedonian lyrics
├── mktl_lyrics.md       # Macedonian transliteration
└── en_lyrics.md         # English translation
```

The Markdown lyric files are treated as source context for labelling. They are not a replacement for checking the generated captions: the labels should still be reviewed for accuracy before training.

## CPU and GPU responsibilities

AceStep work is split across several different workloads:

- **CPU:** audio inspection, the 5Hz LLM's language generation, metadata preparation, and parts of preprocessing.
- **GPU/XPU:** AceStep model execution, VAE encoding/decoding, tensor preprocessing, and LoRA training.
- **System memory:** model staging and offload when the active models cannot all fit in dedicated GPU memory.

The CPU LLM is not automatically a single-threaded Python process. Python orchestration can be serial while the underlying PyTorch or native kernels use multiple threads. The custom labelling path now batches samples and temporarily expands CPU generation threads, but CPU utilisation still depends on the selected model, backend, prompt length, memory bandwidth, and whether the model is actually running on CPU.

Likewise, low GPU utilisation during auto-labelling is expected: that stage is language-model work. GPU utilisation should be judged separately during VAE preprocessing and LoRA training.

## Installation

This fork is intended to be run from the local ComfyUI checkout with the included custom node package. Install the package dependencies in the same Python environment used to launch ComfyUI:

```bash
cd custom_nodes/ComfyUI-FL-AceStep-Training
pip install -r requirements.txt
```

The prebuilt frontend assets are included. Rebuild the frontend only when changing the training widget:

```bash
npm install
npm run build
```

See [`custom_nodes/ComfyUI-FL-AceStep-Training/README.md`](custom_nodes/ComfyUI-FL-AceStep-Training/README.md) for node-level inputs, training parameters, and implementation details.

## Project direction

The goal is a dependable, personal, local AI music workstation:

1. Make multilingual datasets understandable to the labelling model.
2. Make CPU labelling faster without making the workflow opaque.
3. Keep AceStep training practical on the available GPU and system memory.
4. Make every generated label and trained LoRA easy to inspect and reproduce.
5. Use the resulting tools to explore music, language, culture, and model behaviour on Cpntodd's own terms.

## Upstream

This fork builds on [ComfyUI](https://github.com/comfyanonymous/ComfyUI) and [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5). Upstream ComfyUI documentation remains useful for the general node editor and server, but this README intentionally focuses on the AceStep training and local AI work in this fork.
