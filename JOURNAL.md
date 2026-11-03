# Experiment journal

AI renderers for architects promise to keep a design's geometry and materials. How well does that hold, what breaks first, and how can it be measured?

The first test compares two ways of constraining SDXL, ControlNet depth and ControlNet canny, on Blender demo scenes. In a 3D scene the true depth and material of every pixel are known, so results can be checked against ground truth and not only judged by eye.

I started this project on 2 November 2026. Newest entries go at the bottom, failures included.

## 2–3 November 2026: pipeline and first run

### Setup

| | |
|---|---|
| Hardware | Laptop with RTX 3070 (8 GB VRAM), 14 GB RAM |
| Software | ComfyUI 0.43.2, PyTorch 2.14.1 (CUDA 12.6), Blender 5.0.1 |
| Models | SDXL base 1.0, ControlNet Union SDXL promax (xinsir) |
| Frames | 6 views, 1216×832, Cycles with 128 samples |
| Run | 6 frames × 2 methods × 3 seeds = 36 images, same prompts and seeds for both methods |
| Sampling | 25 steps, dpmpp_2m with karras, CFG 6.5, ControlNet strength 0.8, denoise 1 |

Scenes: Classroom by Christophe Seux (CC0), Barcelona Pavillion by eMirage (CC-BY), Italian Flat by Flavio Della Tommasa (CC-BY), all from the Blender demo files.

Inputs: depth comes from the true distance to the camera, converted to inverse depth. Canny edges come from the Blender render, with thresholds 0.2 and 0.5.

In this baseline the model starts from pure noise. The Blender render reaches it only as edges or depth, so the text prompt is the only thing that tells it about materials.

### What went wrong before the first image

1. **Scene downloads.** Two links from the blender.org demo page saved a 48 KB "Thanks!" HTML page under a `.zip` name. Direct links on download.blender.org work.
2. **No passes file.** Rendering to a relative path left no EXR in the expected folder and raised no error. An isolated test confirmed it, so the path is now absolute.
3. **Wrong view layer.** In the Barcelona file `bpy.context.view_layer` pointed at a disabled layer, so the material pass was switched on where nothing renders. The script now uses the first enabled layer.
4. **Pass name.** Blender 5 calls the channel `Material Index.X`, not `IndexMA.X` as I expected.
5. **Garbled depth image.** Writing a two-dimensional array through OpenImageIO put the picture in the left two thirds of the file and noise in the rest. Adding a channel axis fixed it.
6. **Black depth maps on the flat scene.** The artist's cameras use `clip_start` of 3.1 m and 4 m to see past nearby walls. The Depth pass reported values as low as 0.01 m there, and in inverse depth that thin strip stretched the range until the rest of the image went almost black. I switched to computing depth from the Position pass.
7. **A bug from that fix.** Classroom depth came out 1.56 times too large because its camera object has a scale of 0.641. Normalising the camera rotation fixed it.

How I know depth is right now: on the four classroom and Barcelona frames it matches the Depth pass, with the 1st and 99th percentile of the difference at 0.000. On the two flat frames the nearest point is 3.112 m and 4.01 m, just past each camera's clip plane.

### First results, by eye

Nothing is measured yet, so these are impressions to test.

- **Layout holds with both methods.** The main furniture and walls stay in place across the 36 images.
- **Canny keeps flat details that depth cannot see.** Floor joints, chalkboards, notice boards and window frames follow the source. With depth the classroom gains an extra window in one image and a second chalkboard in another, and the Barcelona floor gets invented joints.
- **Depth loses glass.** On the Barcelona terrace the glass wall became curtains, frosted panels or slats in all three depth images and stayed glass in all three canny images.
- **Neither method keeps materials.**
  - Red marble from the wall's description spread onto the floor in all six Barcelona facade images.
  - The weathered stone floor on the terrace became smooth and polished in all six.
  - The cork board in `classroom_left` never came out as cork. It became a chalkboard in five images and a blank wall in one.
- **Canny can invent objects.** One terrace image has a third seat in the foreground, and one kitchen image has a ghost of extra table. My guess is that stray edges get read as furniture.

An earlier test image made with canny had flat, cut-out trees, so I expected depth to handle foliage better. This run supports that only weakly: one of the three canny terrace images shows smeared trees and the other two look fine.

### Hardware

- The 36 images took 32.7 minutes, not the 26 I estimated from a test run.
- Render time rose from 38 s to about 65 s per image during the run while the GPU sat at 88–89 °C. It looks like thermal throttling, but I did not log clock speeds, so it is not confirmed.
- Peak VRAM was 7.1 GB of 8 GB. Free RAM usually stayed near 2 GB and dropped to 0.9 GB once.

### My mistakes in this run

I wrote the prompts from low-resolution previews and got two colours wrong: "green chalkboard" (it is dark slate) and "dark blue armchair" (it is charcoal). The model followed the words. Both methods got the same wrong prompts, so the comparison between them still holds, but colour error against the source will be inflated on those frames. I kept the baseline as it ran and will correct the prompts for the next run.

### Open questions

- Why does the Depth pass disagree with world positions on the flat scene and not on Barcelona? With the same camera tests, the median difference on the flat scene is close to `clip_start` (0.099 at 0.1, 1.88 at 2.0, 3.04 at 3.1), while on Barcelona it is 0.000 at both 0.1 and 2.0. I did not find the cause. Position-based depth avoids the problem, so I moved on.
- Is the slowdown thermal? The next run should log GPU clocks.

### Next

1. Write the metrics script: contour match, depth error against the true depth, colour shift per material mask.
2. Run it on this baseline to replace the impressions above with numbers.
3. Try one fix and measure it. The baseline never shows the model the source colours, so the first candidate is starting from the Blender render (img2img) instead of noise, with the prompts corrected.

## 3 November 2026: trying it by hand in ComfyUI

Until now the images came from a script. Today I opened the same chain in the ComfyUI interface and ran the Barcelona terrace frame by hand, changing one thing at a time. That gave 16 images.

| Image | Strength | Seed | What changed |
|---|---|---|---|
| canny 1 | 0.8 | 1 | nothing, the starting point |
| canny 2 | 0.3 | 1 | strength |
| canny 3 | 1.0 | 1 | strength |
| canny 4 | 1.0 | 1 | floor wording: `warm oak wooden floor` |
| canny 5–11 | 1.0 | 1 | seven wordings for a matte stone floor |
| canny 12 | 0.8 | 2 | seed, original wording |
| canny 13 | 0.8 | 3 | seed, original wording |
| depth 1 | 0.8 | 1 | depth instead of canny, original wording |
| canny 14 | 0.8 | 3 | added `glossy, polished, reflective floor` to the negative prompt |
| canny 15 | 0.8 | 1 | same negative prompt, seed back to 1 |

### What I saw

- **Starting point.** The lighting and the colour of the glass behind the chairs changed, but overall the frame feels improved.
- **Strength 0.3.** The floor became glossy, the chairs look different, an extra chair appeared in the background, and there are extra windows.
- **Strength 1.0.** The picture looks closer to the original, with fewer extra objects, but the floor suffers: it has extra lines and is glossier.
- **Floor wording.** I could not fully fix the floor. The words I used: `on a matte gray flat stone tiles, same-sized, with matte dark concrete-like lines at the border of tiles`.
- **Seed 2.** The colours became more saturated. The floor stands out most: it became lighter and even glossier.
- **Seed 3.** An extra armchair appeared, but the colours are closer to the original and the floor is more matte.
- **Depth, seed 1.** The glass wall became a kind of white curtain, and the floor became glossier.
- **Negative prompt.** I first forgot to change seed 3 back, and a strange structure made of chairs appeared. With the seed changed, the normal floor came back. It seems less glossy, but it is still not really matte and far from ideal.

### Repeatability check

Four of these images used the same settings as the baseline run: canny with seeds 1, 2 and 3, and depth with seed 1. Each one matches its baseline image exactly, with 100% of pixels identical, although one was made through the interface and the other through the API. So on this laptop the same settings give the same image.

## 3 November 2026: first numbers

`scripts/evaluate.py` compares every image with its source frame. It works on images that already exist, so nothing was re-rendered.

| Score | What it asks | Better |
|---|---|---|
| `edge_recall` | Of the true object and material boundaries from Blender, how many still have an edge within 3 pixels? | higher |
| `edge_extra` | Of the edges in the image, how many have no boundary or source edge within 3 pixels? | lower |
| `depth_error` | How far is depth estimated from the image (Depth Anything V2 Small) from the true depth, after fitting scale and shift? | lower |
| `color_shift` | How far is each material's mean colour from the source (CIE76 ΔE, about 2 is barely visible), weighted by area? | lower |
| `chroma_shift` | The same, leaving out lightness | lower |

Glass is left out of everything except `edge_recall`, because it shows whatever is behind it. The first version of `depth_error` included glass and rewarded depth for turning a glass wall into a curtain, since the curtain sits where the true glass plane is. Dense foliage and noise are also left out, 2–12% of each frame.

### Baseline: 36 images

| Score | Source render | Canny | Depth | Canny better in |
|---|---|---|---|---|
| `edge_recall` | 0.876 | 0.838 | 0.811 | 4 of 6 frames |
| `edge_extra` | 0 | 0.311 | 0.373 | 5 of 6 frames |
| `depth_error` | 0.062 | 0.088 | 0.086 | 3 of 6 frames |
| `color_shift` | 0 | 17.2 | 20.3 | 4 of 6 frames |
| `chroma_shift` | 0 | 6.2 | 6.6 | 4 of 6 frames |

How the impressions from the first run held up:

- **Layout holds.** Supported. Depth error is only a little above the 0.062 the estimator gets on the source render itself.
- **Canny keeps flat details better.** Only partly. Canny wins `edge_recall` clearly on `classroom_main` (0.836 against 0.739) and `flat_kitchen` (0.868 against 0.793), but depth wins on `barcelona_facade` (0.851 against 0.803) and `flat_living` (0.855 against 0.833). I did not expect that and have not looked into why.
- **Neither keeps materials.** Supported. Colour shift is 17–20 on average, where about 2 is barely visible. Hue and saturation move less, about 6. The rest is lightness: the terrace, for example, comes out far brighter than the source.
- **Canny can invent objects.** True, but depth adds more unfounded edges overall.
- **New:** `depth_error` does not separate the methods at all.

Between seeds of the same method on the same frame, scores differ on average by 0.044 for `edge_recall`, 0.078 for `edge_extra`, 0.038 for `depth_error` and 5.4 for `color_shift`. That is as large as most differences between the methods, so with 6 frames and 3 seeds I read these as tendencies, not proof.

### My 16 terrace images

Lightness spread is the gap between the darkest and brightest parts of the floor (5th to 95th percentile). It is a rough stand-in for gloss, because reflections widen it.

| Image | What changed | `edge_extra` | `depth_error` | Floor ΔE | Floor lightness spread |
|---|---|---|---|---|---|
| source | | 0 | 0.029 | 0 | 28.9 |
| canny 1 | starting point | 0.406 | 0.046 | 26.5 | 56.2 |
| canny 2 | strength 0.3 | 0.626 | 0.094 | 22.4 | 62.7 |
| canny 3 | strength 1.0 | 0.379 | 0.040 | 28.9 | 54.8 |
| canny 4 | oak floor | 0.441 | 0.055 | 33.7 | 51.2 |
| canny 9 | my final stone wording, strength 1.0 | 0.315 | 0.042 | 20.8 | 47.6 |
| canny 13 | seed 3 | 0.371 | 0.063 | 23.7 | 49.0 |
| canny 14 | seed 3, negative prompt | 0.437 | 0.160 | 20.1 | 76.4 |
| canny 15 | seed 1, negative prompt | 0.364 | 0.041 | 26.2 | 49.8 |
| depth 1 | depth instead of canny | 0.404 | 0.034 | 28.6 | 56.6 |

What the numbers say about what I saw:

- **Strength.** At 0.3 extra edges and depth error both jump, which fits the extra chair and windows. At 1.0 both are the lowest of the three, which fits "closer to the original".
- **Extra furniture.** The extra armchair at seed 3 raises depth error from 0.046 to 0.063, and the chair structure in canny 14 reaches 0.160, the highest of all 16 images.
- **Floor wording.** From canny 3 to canny 9 the floor moved closer to the source on colour (28.9 to 20.8) and got more even (54.8 to 47.6). That is a real improvement, but not a fix.
- **Negative prompt.** From canny 1 to canny 15 the floor colour barely changed (26.5 to 26.2) but it got more even (56.2 to 49.8). That fits "less glossy, but still not really matte": the source is at 28.9.
- **Oak floor.** The chroma shift on the floor jumps to 15.9, against 3.1 for canny 3 with the same strength and seed.

### Limits of these scores

- `edge_extra` also counts real details when the source is too dark to show them, such as the cushion tufting on the terrace chairs.
- On the classroom the estimator's own error on the source render is 0.11–0.17, so `depth_error` says little there.
- Mean colour does not see texture. Lightness spread also rises when something else covers the floor, as in canny 14.
- The wrong colours in two prompts (see above) inflate colour scores on the classroom and living room for both methods.

### Next

Try the first real fix and measure it with this script: start from the Blender render (img2img) instead of noise, with the prompts corrected.

## 3 November 2026: starting from the render

The baseline never shows the model the source colours. So I changed one thing: generation now starts from the Blender render with noise added, instead of from pure noise. The `denoise` setting decides how much noise: close to 0 returns the render almost unchanged, and 1 is the baseline.

I also added a score, `pixel_change`: the mean colour difference per pixel from the source. It shows how much the model changed at all, and it reacts to new texture, which the mean colour per material does not.

### Choosing the value by hand

I ran the terrace frame in ComfyUI with canny, strength 0.8 and seed 1, changing only `denoise`.

| `denoise` | `edge_extra` | `depth_error` | `color_shift` | Floor ΔE | Floor lightness spread | `pixel_change` |
|---|---|---|---|---|---|---|
| 1.0, from noise | 0.406 | 0.046 | 26.4 | 26.5 | 56.2 | 31.3 |
| 0.8 | 0.349 | 0.048 | 14.7 | 15.2 | 49.0 | 16.7 |
| 0.6 | 0.266 | 0.039 | 3.4 | 2.8 | 31.5 | 7.3 |
| 0.5 | 0.126 | 0.034 | 1.9 | 1.4 | 28.7 | 5.6 |
| 0.4 | 0.090 | 0.031 | 1.1 | 0.6 | 28.3 | 4.8 |
| source | 0 | 0.029 | 0 | 0 | 28.9 | 0 |

What I saw:

- **0.8.** Worse than 0.6. The column is round, there are no hatches, but the windows look strange and the floor is glossier.
- **0.6.** The column on the left became round and extra hatches appeared on the ceiling. The windows and chairs look decent. The floor is less glossy, but it has random lines.
- **0.5.** I think it looks more like a photo than a 3D render, but 0.4 looked even more like one. Depth feels stronger at 0.5. The floor is acceptable and the column, chairs and glass look good, but to me the colour is a bit dark.
- **0.4.** Closer to the original. The column is no longer round, the hatches are gone, and the floor is practically identical.

I chose 0.4. The numbers agree with most of this. One difference: I found 0.5 a bit dark, but its lightness is as close to the source as that of 0.4 (+0.4 and +0.2). The darkness comes from the source render, which the model no longer brightens.

The first image from the script at 0.4 matched my hand-run image exactly, with 100% of pixels identical.

### All frames at 0.4: 36 images

Same frames, seeds, prompts and strength as the baseline. Only the starting point changed.

| Score | Source render | Canny, from noise | Canny, 0.4 | Depth, from noise | Depth, 0.4 |
|---|---|---|---|---|---|
| `edge_recall` | 0.876 | 0.838 | 0.849 | 0.811 | 0.863 |
| `edge_extra` | 0 | 0.311 | 0.071 | 0.373 | 0.076 |
| `depth_error` | 0.062 | 0.088 | 0.073 | 0.086 | 0.073 |
| `color_shift` | 0 | 17.2 | 1.2 | 20.3 | 1.2 |
| `chroma_shift` | 0 | 6.2 | 1.0 | 6.6 | 1.0 |
| `pixel_change` | 0 | 24.1 | 4.6 | 28.5 | 4.8 |

What got better:

- **Materials.** `color_shift` fell below the level of about 2 that is barely visible, in all 6 frames with both methods. The red marble no longer spreads over the facade floor: its chroma shift there fell from 11.9 to 1.1. The cork board in `classroom_left` stays cork: its ΔE fell from 25.8 to 1.3.
- **Invented lines.** `edge_extra` fell in all 6 frames with both methods.
- **Stability.** The difference between seeds fell from 5.4 to 0.33 on `color_shift` and from 0.044 to 0.017 on `edge_recall`.

What did not get better:

- `edge_recall` fell on the terrace with canny (0.866 to 0.848) and on the facade with depth (0.851 to 0.832).
- `depth_error` rose on `classroom_left` with both methods and on `classroom_main` with depth. The estimator's own error on those source renders is 0.17 and 0.11, so I would not read much into this.
- Colour is not equally good everywhere. 17 materials are still above 2: tree trunks on the terrace are the worst at 10.6, and the others are at most 5.

**Canny against depth.** The gap from the baseline mostly closes. Colour is equal, and `edge_extra` is 0.071 against 0.076. `edge_recall` now goes the other way: depth is better in all 6 frames, 0.863 against 0.849. At this setting the choice between them matters little.

### Corrected prompts: 18 images

After the main run I corrected the two wrong colours and re-ran only the three frames they affect. The baseline and the main run at 0.4 used the prompts as of commit 6597b2e.

| Frame | `color_shift`, wrong prompt | `color_shift`, corrected |
|---|---|---|
| `classroom_main` | 1.59 | 1.43 |
| `classroom_left` | 1.27 | 1.12 |
| `flat_living` | 0.88 | 0.86 |

The chalkboard in `classroom_left` improved most, from a ΔE of 1.88 to 0.94. Geometry scores barely moved. Starting from the render, a wrong colour in the prompt does little harm.

### The slowdown was thermal

This answers the open question from the first run. The new log shows the GPU clock falling from 1697 MHz at the start to about 950 MHz, with the thermal slowdown flag on in nearly every sample while the GPU was busy, at up to 89 °C. Each image took 36 s at the start and about 47 s later on, and the 36 images took 28 minutes. Peak VRAM was 7.1 GB of 8 GB. It costs time only: the script's image, made on a hot GPU, still matched the hand-run image exactly.

### Limits of this result

- **The scores reward closeness to the source.** At `denoise` near 0 they would be almost perfect while nothing had improved. `pixel_change` fell from about 26 to 4.7, so a large part of the gain is simply that the model changes much less. None of the scores says whether an image looks more like a photograph. So far that is only my judgement by eye.
- **Mean colour does not see texture.** On the facade the weathered stone floor with dark specks became smoother, veined stone, although its mean colour moved by only 1.1. Its `pixel_change` of 5.2 is higher than the 2.3 on the white ceiling, but this score alone does not separate texture from other change.
- **One value for everything.** I chose 0.4 on one frame and one seed.
- **The terrace stays as dark as the source.** Starting from the render keeps its lighting too, so this setting will not fix a dull render.

### Next

Show it: a README and a before and after page with the results and the failures.
