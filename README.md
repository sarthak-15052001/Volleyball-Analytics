# Volleyball Analytics

Event **classification & player association** for volleyball match videos (YOLO11, ByteTrack, ball tracking, PaddleOCR jerseys).

## Layout

| Path | Purpose |
|------|---------|
| `train/` | Match videos (training) |
| `holdout/` | Unseen test videos |
| `uploads/` | Videos uploaded from the notebook |
| `Volleyball_YOLO_Player_Detection_Colab.ipynb` | Main notebook |
| `volleyball_analytics/pipeline.py` | Event + jersey pipeline |
| `export_analytics.py` | CLI export to `o/p/` |
| `models/` | Fine-tuned player weights (optional) |
| `o/p/events_csv/` | Per-video event CSVs with time ranges |

## Setup (Ubuntu, conda `py312`)

```bash
conda activate py312
cd /home/iflair/Documents/Volleyball-Anaylitics
pip install -r requirements-analytics.txt
```

In the notebook: run **§1 Environment setup**, restart the kernel, then run from **§2** onward.

## Quick test

- **Notebook §13**: first video, `SMOKE_MAX_SECONDS` (default 90s).
- **Notebook §14**: upload any video and process.
- **CLI**:

```bash
python export_analytics.py --max-seconds 90 --limit-videos 1
python export_analytics.py --video /path/to/match.mp4 --max-seconds 120
python export_analytics.py --full
```

## CSV columns (events)

Each row is one event segment: `time_start_sec`, `time_end_sec`, `timestamp_start`, `timestamp_end`, `player_track_id`, `jersey_number`, `rally_id`, `touch_index`, `ball_speed_peak`.

## Hardware (RTX 3050 6 GB)

- `yolo11n.pt` for players and ball (COCO sports ball until you add a fine-tuned ball `.pt`)
- PaddleOCR on CPU to save VRAM
- Inference every 2nd frame (`infer_stride=2`)

## Next accuracy steps

- Fine-tune YOLO11 on volleyball **ball** and **jersey digit** crops
- Label event clips and train VideoMAE / X3D (hook point: replace `RallyEventEngine` classifier)
