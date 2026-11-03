#!/usr/bin/env bash
# Render every benchmark frame from the three demo scenes. Run from the repository root.
# Set BLENDER to use another Blender 5 install.
set -euo pipefail

BLENDER="${BLENDER:-/c/Program Files/Blender Foundation/Blender 5.0/blender.exe}"

render() { # scene id, .blend file, extra Blender arguments
    "$BLENDER" -b --factory-startup --disable-autoexec "$2" "${@:3}" \
        --python-exit-code 1 --python scripts/render_frames.py -- \
        --views config/views.json --scene-id "$1" --out frames
}

render classroom scenes/classroom/classroom/classroom.blend
render barcelona scenes/barcelona/3d/pavillon_barcelone_v1.2.blend -S "1- time: midday"
render flat scenes/flat/flat-archiviz.blend
