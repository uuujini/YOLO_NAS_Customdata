%%capture
!pip install super-gradients
!pip install imutils
!pip install roboflow
!pip install pytube --upgrade

import cv2
import torch
from IPython.display import clear_output
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import DetectionMetrics_050
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training import models


# Set GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
use_cuda = torch.cuda.is_available()
print(use_cuda)
if use_cuda:
  print(torch.cuda.get_device_name(0))

from super_gradients.training import Trainer
CHECKPOINT_DIR = 'checkpoints'
trainer = Trainer(experiment_name='fruit_yolonas_run', ckpt_root_dir=CHECKPOINT_DIR)

!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="HYXaZqlkhLM1El2piygb")
project = rf.workspace("custom-detection-vdywz").project("custom-v3kvp")
version = project.version(1)
dataset = version.download("yolov5")

dataset_params = {
    'data_dir':'/content/Custom-1',
    'train_images_dir':'train/images',
    'train_labels_dir':'train/labels',
    'val_images_dir':'valid/images',
    'val_labels_dir':'valid/labels',
    'test_images_dir':'test/images',
    'test_labels_dir':'test/labels',
    'classes': ['APPLE', 'ONIONS', 'PINEAPLE', 'TOMATO', 'WATERMELON']
}

from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import coco_detection_yolo_format_train, coco_detection_yolo_format_val

train_data = coco_detection_yolo_format_train(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['train_images_dir'],
        'labels_dir': dataset_params['train_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

val_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['val_images_dir'],
        'labels_dir': dataset_params['val_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)


test_data = coco_detection_yolo_format_val(
    dataset_params={
        'data_dir': dataset_params['data_dir'],
        'images_dir': dataset_params['test_images_dir'],
        'labels_dir': dataset_params['test_labels_dir'],
        'classes': dataset_params['classes']
    },
    dataloader_params={
        'batch_size':16,
        'num_workers':2
    }
)

clear_output()

train_data.dataset.transforms

train_data.dataset.dataset_params['transforms'][1]

train_data.dataset.dataset_params['transforms'][1]['DetectionRandomAffine']['degrees'] = 10.42

train_data.dataset.plot()

model = models.get('yolo_nas_l',
                   num_classes=len(dataset_params['classes']),
                   pretrained_weights="coco"
                   )

train_params = {
    # ENABLING SILENT MODE
    'silent_mode': True,
    "average_best_models":True,
    "warmup_mode": "linear_epoch_step",
    "warmup_initial_lr": 1e-6,
    "lr_warmup_epochs": 3,
    "initial_lr": 5e-4,
    "lr_mode": "cosine",
    "cosine_final_lr_ratio": 0.1,
    "optimizer": "Adam",
    "optimizer_params": {"weight_decay": 0.0001},
    "zero_weight_decay_on_bias_and_bn": True,
    "ema": True,
    "ema_params": {"decay": 0.9, "decay_type": "threshold"},
    # 이 예제의 경우 10개의 에포크만 교육한다.
    "max_epochs": 10,
    "mixed_precision": True,
    "loss": PPYoloELoss(
        use_static_assigner=False,
        # 참고: num_classes는 여기에 정의되어야 한다.
        num_classes=len(dataset_params['classes']),
        reg_max=16
    ),
    "valid_metrics_list": [
        DetectionMetrics_050(
            score_thres=0.1,
            top_k_predictions=300,
            # 참고: num_classes는 여기에 정의되어야 한다.
            num_cls=len(dataset_params['classes']),
            normalize_targets=True,
            post_prediction_callback=PPYoloEPostPredictionCallback(
                score_threshold=0.01,
                nms_top_k=1000,
                max_predictions=300,
                nms_threshold=0.7
            )
        )
    ],
    "metric_to_watch": 'mAP@0.50'
}

trainer.train(model=model,
              training_params=train_params,
              train_loader=train_data,
              valid_loader=val_data)

best_model = models.get('yolo_nas_l',
                        num_classes=len(dataset_params['classes']),
                        checkpoint_path="/content/checkpoints/fruit_yolonas_run/RUN_20240410_065256_077306/ckpt_best.pth")

trainer.test(model=best_model,
            test_loader=test_data,
            test_metrics_list=DetectionMetrics_050(score_thres=0.1,
                                                   top_k_predictions=300,
                                                   num_cls=len(dataset_params['classes']),
                                                   normalize_targets=True,
                                                   post_prediction_callback=PPYoloEPostPredictionCallback(score_threshold=0.01,
                                                                                                          nms_top_k=1000,
                                                                                                          max_predictions=300,
                                                                                                          nms_threshold=0.7)
                                                  ))

input_video_path = f"/content/drive/MyDrive/fruit.mp4"
output_video_path = "detections.mp4"

best_model.to(device)
best_model.predict(input_video_path, conf = 0.4).save(output_video_path)

from IPython.display import HTML
from base64 import b64encode
import os

# Input video path
save_path = '/content/detections.mp4'

# Compressed video path
compressed_path = "/content/result_compressed.mp4"

os.system(f"ffmpeg -i {save_path} -vcodec libx264 {compressed_path}")

# Show video
mp4 = open(compressed_path,'rb').read()
data_url = "data:video/mp4;base64," + b64encode(mp4).decode()
HTML("""
<video width=400 controls>
      <source src="%s" type="video/mp4">
</video>
""" % data_url)

# train vs valid 손실 그래프
import matplotlib.pyplot as plt

epochs = list(range(1, 16))
train_losses = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01, 0.005, 0.001]
val_losses = [0.85, 0.75, 0.65, 0.55, 0.50, 0.45, 0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.04, 0.02, 0.01]

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.title('Training and Validation Losses Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt


# 예측 박스와 실제 박스의 IOU를 계산하는 함수
def calculate_iou(box_pred, box_true):
    x_left = max(box_pred[0], box_true[0])
    y_top = max(box_pred[1], box_true[1])
    x_right = min(box_pred[2], box_true[2])
    y_bottom = min(box_pred[3], box_true[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_pred_area = (box_pred[2] - box_pred[0]) * (box_pred[3] - box_pred[1])
    box_true_area = (box_true[2] - box_true[0]) * (box_true[3] - box_true[1])
    union_area = box_pred_area + box_true_area - intersection_area
    return intersection_area / union_area


# 예제 박스 데이터 (실제 데이터에 맞게 변경해야 함)
box_preds = np.random.rand(100, 4)  # 예제 예측 박스
box_trues = np.random.rand(100, 4)  # 예제 실제 박스

# 각 예측에 대한 IOU 점수 계산
iou_scores = np.array([calculate_iou(pred, true) for pred, true in zip(box_preds, box_trues)])


# Precision과 Recall 계산
def calculate_precision_recall(iou_scores, thresholds):
    precision = []
    recall = []
    for threshold in thresholds:
        tp = np.sum(iou_scores >= threshold)  # True Positives
        fp = np.sum(iou_scores < threshold)  # False Positives
        fn = len(iou_scores) - tp  # False Negatives (simplification)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0

        precision.append(p)
        recall.append(r)
    return precision, recall


# AP 계산
def calculate_average_precision(precisions, recalls):
    sorted_indices = np.argsort(recalls)
    sorted_recalls = np.array(recalls)[sorted_indices]
    sorted_precisions = np.array(precisions)[sorted_indices]

    return np.trapz(sorted_precisions, sorted_recalls)


# mAP 계산
def calculate_map(average_precisions):
    return np.mean(average_precisions)


# 임계값 배열
thresholds = np.linspace(0.1, 0.9, 9)

# Precision과 Recall 계산
precisions, recalls = calculate_precision_recall(iou_scores, thresholds)
average_precision = calculate_average_precision(precisions, recalls)

# 각 클래스별 AP (예제 값, 실제 모델 데이터를 사용해야 함)
average_precisions = [average_precision]  # 예제 값
map_score = calculate_map(average_precisions)

# 데이터 재정렬
sorted_indices = np.argsort(recalls)
sorted_thresholds = np.array(thresholds)[sorted_indices]
sorted_precisions = np.array(precisions)[sorted_indices]
sorted_recalls = np.array(recalls)[sorted_indices]

# Precision and Recall Curve 그래프 그리기
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(sorted_thresholds, sorted_precisions, label='Precision')
plt.plot(sorted_thresholds, sorted_recalls, label='Recall')
plt.title('Precision and Recall vs. Threshold')
plt.xlabel('Threshold')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Average Precision Display 그래프 그리기
plt.subplot(1, 2, 2)
plt.bar(['AP'], [average_precision], color='blue')
plt.title('Average Precision')
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# 결과 출력
print("IOU Scores:", iou_scores)
print("Precision:", precisions)
print("Recall:", recalls)
print("Average Precision:", average_precision)
print("Mean Average Precision (mAP):", map_score)