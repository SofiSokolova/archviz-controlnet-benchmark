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
