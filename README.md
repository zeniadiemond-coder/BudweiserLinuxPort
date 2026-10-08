# OpenV6

A C++ library and command-line renderer that synthesizes singing with a VOCALOID6 voice package.

Prerequisites: CMake 3.20+, a C++20 compiler (GCC or Clang) and a VOCALOID6 voice package.

## Build

```sh
cmake -B build
cmake --build build
ctest --test-dir build
```

Pass `-DOPENV6_NATIVE=ON` to build with `-march=native` (not portable).

## Usage

```sh
build/openv6_render <voice-package> <score.json> <out.wav>
```

A voice package is a directory which contains:

- `v6.yaml`: `bank_id`, `language_id` (0 = Japanese, 1 = English, 4 = Chinese) and an optional `model_directory` pointing at the model files if they live elsewhere
- `bank.json`: the bank's registration, looked up by `bank_id` (model names, `NPIdx`, timing condition)
- the models it names: the timbre model `<TModelName>.vtb2`, the pitch model `<PModelName>.vpit` and the timing model `<TModelName>.vtmg` (the last two fall back to `VOCALOID6.vpit` and `VOCALOID6.vtmg`)

Lyrics can be Japanese, English, or Chinese (pinyin). English lyrics use the `G2PAManager_ENG.dll` bundled in `native/`. To use another copy, point `--native-dir` or `$OPENV6_NATIVE` at its directory. Run `openv6_render` without arguments for all score fields and options.

## score.json

```json
{
  "tempoBpm": 120,
  "languageId": 0,
  "notes": [
    {"startTick": 0, "durationTick": 480, "pitch": 67, "lyric": "ど"},
    {"startTick": 480, "durationTick": 480, "pitch": 69, "lyric": "れ",
     "aiParams": {"pitchFine": 0.6}, "vibrato": {"depth": 40}},
    {"startTick": 960, "durationTick": 960, "pitch": 71, "phonemes": "m i"}
  ],
  "controllers": [{"name": "Dynamics", "events": [{"tick": 0, "value": 80}]}]
}
```

Ticks are 480 per quarter note, in 4/4. Property names are case-insensitive.

Required:

- `tempoBpm`: a single constant tempo
- `languageId`: 0 = Japanese, 1 = English, 4 = Chinese
- `notes`: monophonic and non-overlapping, each with `startTick` (absolute, ≥ 0), `durationTick` (> 0) and `pitch` (MIDI note, 0-127)

Optional per note:

- `lyric`: converted to phonemes unless `phonemes` is given. r, rest, sil, pau, sp, ap, cl, っ and ッ make a silent note, and br or 息 makes a breath. "+~" or "+*" extends the vowel of the touching note before it.
- `phonemes`: space-separated phonemes, e.g. "k a"
- `phoneticHint`: phonemes used instead of converting the lyric
- `languageId`: overrides the score's language
- `phonemeTicks`: per phoneme, its start as ticks from the note start, or null for automatic
- `aiParams`: each 0-1, 0.5 neutral: `pitchFine`, `pitchDriftStart`, `pitchDriftEnd`, `pitchScalingCenter`, `pitchScalingOrigin`, `pitchTransitionStart`, `pitchTransitionEnd`, `amplitudeWhole`, `amplitudeStart`, `amplitudeEnd`, `vibratoLeadingDepth`, `vibratoFollowingDepth`
- `vibrato`: `depth` and `rate` (0-127, default 0 and 50), `duration` in ticks back from the note end (default two thirds of the note), `ai` (default true)

Optional per score:

- `controllers`: `[{"name", "events": [{"tick", "value"}]}]` with strictly increasing ticks, each value held until the next. Names: `Expression` (0-100, 50 neutral), `Dynamics` (0-127, 64 neutral), `S5Air` (-100 to 100) and `S5Character` (-64 to 63), both 0 neutral
- `pitch` and `pitchTicks`: equal-length arrays; `pitch` is absolute pitch in cents (MIDI note × 100, A4 = 6900), interpolated linearly between the ticks
- `breaths`: `[{"startTick", "durationTick"}]`
