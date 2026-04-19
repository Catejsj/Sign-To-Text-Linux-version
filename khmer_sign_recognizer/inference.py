"""
Live Inference — Real-time sign language recognition.

Uses the existing dual pipeline (RTMPose body + MediaPipe hands)
with a trained model to predict signs live from the webcam.

Pipeline:
  Camera → dual pipeline → normalizer → sliding window (60 frames)
                                              ↓
                                        model.predict()
                                              ↓
                                     "hello" displayed on screen
"""

import sys
import time
import json
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils import load_config, setup_logging
from capture import LandmarkCapture
from normalizer import FrameNormalizer, TOTAL_JOINTS

# PyTorch — optional, fails gracefully if not installed
try:
    import torch
    from model import SignLanguageCNN
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def load_model(config, logger):
    """Attempt to load the trained model. Returns (model, label_map) or (None, None)."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not installed — running in display-only mode.")
        return None, None

    inf_config   = config.get('inference', {})
    weights_path = inf_config.get('weights_path', 'models/weights/ksl_model_latest.pth')

    if not Path(weights_path).exists():
        logger.warning(f"No weights file at '{weights_path}' — running without model.")
        logger.info("Record data first, train in Colab, then place weights here.")
        return None, None

    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = SignLanguageCNN.load_from_weights(weights_path, device)
    logger.info(f"Model loaded from '{weights_path}' on {device}")

    # Load label map
    labels_path = Path(config.get('recording', {}).get('data_dir', 'data/sequences')) / 'labels.json'
    label_map   = {}
    if labels_path.exists():
        with open(labels_path) as f:
            name_to_idx = json.load(f)
            label_map = {v: k for k, v in name_to_idx.items()}  # idx → name
        logger.info(f"Labels loaded: {list(name_to_idx.keys())}")
    else:
        logger.warning("No labels.json found — predictions will show indices only.")

    return model, label_map


def main():
    config = load_config('config/settings.json')
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("SIGN LANGUAGE LIVE INFERENCE")
    logger.info("=" * 60)

    inf_config   = config.get('inference', {})
    window_size  = inf_config.get('sliding_window', 60)
    interval     = inf_config.get('inference_interval', 15)
    threshold    = inf_config.get('confidence_threshold', 0.6)

    # Initialize components
    capture    = LandmarkCapture(config)
    normalizer = FrameNormalizer(smoothing=True, min_cutoff=1.0, beta=0.05)
    model, label_map = load_model(config, logger)

    if not capture.start():
        logger.error("Failed to start camera!")
        return

    # Sliding window buffer
    frame_buffer = []
    frame_count  = 0
    prediction   = ""
    confidence   = 0.0

    logger.info(f"Running — window={window_size}, interval={interval}, threshold={threshold}")
    logger.info("Press Q to quit")

    try:
        while True:
            ret, frame = capture.read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            # Process landmarks
            landmarks = capture.process_landmarks(frame)

            if landmarks:
                # Normalize
                clean, _ = normalizer.normalize(
                    landmarks.get('pose', {}),
                    landmarks.get('left_hand', {}),
                    landmarks.get('right_hand', {}),
                    landmarks.get('image_width', 640),
                    landmarks.get('image_height', 480),
                    time.time()
                )

                frame_buffer.append(clean)

                # Keep only latest window_size frames
                if len(frame_buffer) > window_size:
                    frame_buffer = frame_buffer[-window_size:]

                frame_count += 1

                # Run inference every N frames
                if (model is not None and
                    len(frame_buffer) == window_size and
                    frame_count % interval == 0):

                    prediction, confidence = _predict(
                        model, frame_buffer, label_map
                    )

            # ── Draw UI ──
            h, w = frame.shape[:2]

            # Prediction display
            if model is not None:
                if confidence >= threshold and prediction:
                    # High confidence — green
                    cv2.putText(frame, prediction.upper(),
                                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.5, (0, 255, 0), 3)
                    cv2.putText(frame, f"{confidence:.0%}",
                                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                elif prediction:
                    # Low confidence — yellow
                    cv2.putText(frame, f"({prediction}?)",
                                (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 255, 255), 2)
                    cv2.putText(frame, f"{confidence:.0%}",
                                (10, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 255), 2)

                # Buffer fill indicator
                fill = len(frame_buffer) / window_size
                bar_w = 150
                cv2.rectangle(frame, (w - bar_w - 10, 10),
                              (w - 10, 25), (80, 80, 80), -1)
                cv2.rectangle(frame, (w - bar_w - 10, 10),
                              (w - bar_w - 10 + int(bar_w * fill), 25),
                              (0, 200, 0), -1)
            else:
                cv2.putText(frame, "NO MODEL LOADED",
                            (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2)

            # Status
            cv2.putText(frame, "INFERENCE MODE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 200, 0), 2)

            # Resize
            display_frame = frame
            if 'window_scale' in config['capture']:
                scale = config['capture']['window_scale']
                if scale != 1.0:
                    new_w = int(frame.shape[1] * scale)
                    new_h = int(frame.shape[0] * scale)
                    display_frame = cv2.resize(frame, (new_w, new_h))

            cv2.namedWindow('Sign Language Inference', cv2.WINDOW_NORMAL)
            cv2.imshow('Sign Language Inference', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")

    finally:
        capture.stop()
        cv2.destroyAllWindows()
        logger.info("Inference stopped.")


def _predict(model, frame_buffer, label_map):
    """Run inference on the frame buffer."""
    # Stack into (1, seq_len, 51, 3)
    seq = np.array(frame_buffer, dtype=np.float32)
    tensor = torch.from_numpy(seq).unsqueeze(0)

    if next(model.parameters()).is_cuda:
        tensor = tensor.cuda()

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

    class_idx  = idx.item()
    confidence = conf.item()
    label_name = label_map.get(class_idx, f"class_{class_idx}")

    return label_name, confidence


if __name__ == "__main__":
    main()
