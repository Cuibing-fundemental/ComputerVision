import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

def show_frames(cap,frame_indices):
    max_frames = 9    
    frame_indices = frame_indices[:max_frames]
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for i, frame in enumerate(frames):
        axes[i].imshow(frame)
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

def train():
    model = YOLO("yolov8s.pt")
    # print(model.model)

    model.train(
        data="./data/VisDrone/new_data.yaml",
        epochs=100,
        imgsz=960,
        batch=16,
        device="cuda",
        optimizer="AdamW",
        lr0 = 1e-4,
        workers=4,
        patience=10,

        project="./yolo_output",
        name="visdrone_exp1",
    )

def test():
    model = YOLO('./model/best_yolov8s.pt')

    results = model.track(
        source='./data/test_video/input.mp4',
        tracker="bytetrack.yaml",
        persist=True,
        save=True,
        stream=True,
        show=False,
        name="output"
    )

    for r in results:
        pass

    print(model.predictor.save_dir)

def count():
    model =  YOLO('./model/best_yolov8s.pt')
    video_path = './data/test_video/input.mp4'

    results = model.track(
        source=video_path,
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    line_x = width // 3
    previous_positions = {}
    counted_ids = set()

    counter = 0
    idx = 0
    for r in results:
        frame = r.orig_img
        cv2.line(
            frame,
            (line_x, 0),
            (line_x, height),
            (0, 0, 255),
            2
        )

        boxes = r.boxes
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            xyxys = boxes.xyxy.cpu().numpy()
            for track_id, box in zip(ids, xyxys):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                if track_id in previous_positions:
                    prev_cx = previous_positions[track_id]
                    if (
                        (prev_cx < line_x and cx >= line_x) or (cx < line_x and prev_cx >= line_x)
                        and track_id not in counted_ids
                    ):
                        counter += 1
                        counted_ids.add(track_id)
                        cls_id = int(boxes.cls.cpu().numpy()[0])
                        print(f"ID: {track_id}, frame: {idx}, position: ({cx}, {cy}), label: {model.names[cls_id]}")
                previous_positions[track_id] = cx
        idx += 1
    print("-" * 30)
    print(f"Total count: {counter}")

if __name__ == "__main__":
    # train()
    # test() 
    count()