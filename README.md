# Real-Time Lip Reading Assistant

Accessibility-focused visual speech recognition prototype. It uses webcam video, MediaPipe Face Mesh, mouth crops, and a small PyTorch CNN+GRU model to classify isolated spoken words from mouth movement only.

This is intentionally small and interview-friendly. It is not full sentence lip reading, does not use audio, and is not a LipNet clone.

## Vocabulary

- `hello`
- `yes`
- `no`
- `help`
- `stop`
- `water`
- `food`
- `medicine`
- `doctor`
- `pain`
- `emergency`
- `call`
- `thank_you`
- `please`
- `bathroom`
- `goodbye`
- `repeat`
- `slow_down`
- `more`
- `finished`
- `where`
- `when`
- `who`
- `family`

## Architecture

```text
Webcam -> Face Mesh -> Mouth Crop -> CNN Frame Encoder -> GRU -> Word Prediction
```

The model treats lip reading as a temporal computer vision task. One mouth image is not enough; the model receives a short clip, usually 30 frames, so it can learn motion across time.

## Setup

```bash
cd lip-reading-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

## Collect Data

Record fixed-length mouth clips for each word:

```bash
python collect_data.py --label hello --num_samples 30 --frames_per_clip 30
python collect_data.py --label yes --num_samples 30 --frames_per_clip 30
python collect_data.py --label no --num_samples 30 --frames_per_clip 30
```

Controls:

- Press `SPACE` to record one sample.
- Press `Q` to quit.

Clips save as `.npy` files under:

```text
data/raw/<label>/
```

Suggested recording plan:

- Start with 20 to 50 samples per word.
- Record in the same lighting first.
- Add different lighting, camera distance, and slight head angles later.
- Keep clip length consistent between collection, training, and inference.

## Train

```bash
python train.py --data_dir data/raw --epochs 15 --batch_size 8 --lr 0.001 --frames_per_clip 30 --img_size 96
```

The script automatically uses CUDA, Apple Silicon MPS, or CPU. Best checkpoint saves to:

```text
models/lip_reader.pt
```

Training prints:

- training loss
- training accuracy
- validation loss
- validation accuracy

## Live Demo

```bash
python lip_reader.py --checkpoint models/lip_reader.pt
```

The demo overlays:

- predicted word
- confidence
- FPS
- rolling buffer status
- missing-checkpoint warning

If no checkpoint exists, the app still opens webcam and runs mouth detection demo mode.

Press `Q` to quit.

## Files

```text
collect_data.py  webcam data collection
train.py         dataset loading, training, checkpoint saving
lip_reader.py    real-time webcam inference
model.py         CNN+GRU PyTorch model
utils.py         Face Mesh ROI, preprocessing, labels, FPS helpers
```

## Limitations

- Isolated word classification only.
- Small vocabulary.
- Accuracy depends heavily on your collected data.
- Single-speaker data may not generalize to other speakers.
- Similar mouth shapes can be confused, especially without audio.
- Real-world accessibility use would require more data, testing, and robust UI design.

## Future Improvements

- Larger dataset with more speakers.
- Sentence-level recognition.
- CTC loss for unsegmented sequences.
- Transformer or 3D CNN temporal model.
- Better augmentation for lighting and camera angle changes.
- Integration with captions or assistive communication UI.
