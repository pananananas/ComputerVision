# ## Laboratorium 7
# Detekcja obiektów za pomocą Faster-RCNN

# ### Wprowadzenie
# 
# Celem tej listy jest praktyczne zapoznanie się z działaniem dwuetapowych modeli do detekcji obiektów na przykładzie Faster R-CNN. Skorzystamy z gotowej implementacji modelu z pakietu [`torchvision`](https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py). Jeżeli masz inny ulubiony model działający na podobnej zasadzie, możesz z niego skorzystać zamiast podanego. Podobnie implementacja - jeśli masz swoją ulubioną bibliotekę np. Detectron2, MMDetection, możesz z niej skorzystać.
# 
# W zadaniu wykorzystany zostanie zbiór danych [_Chess Pieces Dataset_](https://public.roboflow.com/object-detection/chess-full) (autorstwa Roboflow, domena publiczna), ZIP z obrazami i anotacjami powinien być dołączony do instrukcji.
# 
# Podczas realizacji tej listy większy nacisk położony zostanie na inferencję z użyciem Faster R-CNN niż na uczenie (które przeprowadzisz raz\*). Kluczowe komponenty w tej architekturze (RPN i RoIHeads) można konfigurować bez ponownego uczenia, dlatego badania skupią się na ich strojeniu. Aby zrozumieć działanie modelu, konieczne będzie spojrzenie w jego głąb, włącznie z częściowym wykonaniem. W tym celu warto mieć na podorędziu kod źródłowy, w szczególności implementacje następujących klas (uwaga - linki do najnowszej implementacji; upewnij się więc, że czytasz kod używanej przez siebie wersji biblioteki):
# * `FasterRCNN`: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/faster_rcnn.py
# * `GeneralizedRCNN`: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/generalized_rcnn.py
# * `RegionProposalNetwork`: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/rpn.py
# * `RoIHeads`: https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
# 
# Dogłębne zrozumienie procedury uczenia modelu nie będzie wymagane, niemniej należy mieć ogólną świadomość jak ten proces przebiega i jakie funkcje kosztu są wykorzystywane. Użyjemy gotowej implementacji z submodułu [`references.detection`](https://github.com/pytorch/vision/blob/main/references/detection/train.py) w nieco uproszczonej wersji. Ponieważ ten moduł **nie** jest domyślnie instalowaną częścią pakietu `torchvision`, do instrukcji dołączono jego kod w nieznacznie zmodyfikowanej wersji (`references_detection.zip`).
# Jeśli ciekawią Cię szczegóły procesu uczenia, zachęcam do lektury [artykułu](https://arxiv.org/abs/1506.01497) i analizy kodu implementacji.


# ### Zadanie 0: Uczenie
# 
# Krokiem "zerowym" będzie przygotowanie wstępnie nauczonego modelu i douczenie go na docelowym zbiorze.
# Podany zestaw hiperparametrów powinien dawać przyzwoite (niekoniecznie idealne) wyniki - jeśli chcesz, śmiało dobierz swoje własne; nie spędzaj na tym jednak zbyt wiele czasu.
# 
# Twoim zadaniem jest nie tylko przeklikanie poniższych komórek, ale przynajmniej ogólne zrozumienie procesu uczenia (przejrzyj implementację `train_one_epoch`) i struktury modelu.

import os
import time
import torch
import datetime
import numpy as np
import torch.utils.data
import matplotlib.pyplot as plt
import torchvision.models.detection as M
from torchvision.io.image import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms.functional import to_pil_image
from detection import coco_utils, presets, utils, transforms
from vis import visualize_rpn_nms_analysis, visualize_timing_analysis, create_heatmap, visualize_proposals_and_predictions, plot_precision_recall_curves
from detection.engine import train_one_epoch, evaluate
from collections import OrderedDict
from pathlib import Path
from tqdm import tqdm

def set_seed(seed=42):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def get_dataset(img_root:str, file_name:str, train:bool=True):
    """Reimplementacja analogicznej funkcji z pakietu references, rozwiązująca drobną niekompatybilność w zbiorze CPD"""
    def fake_segmentation(image, target):
        for obj in target['annotations']:
            x, y, w, h = obj['bbox']
            segm = [x, y, x+w, y, x+w, y+h, x, y+h]
            obj['segmentation'] = [segm]
        return image, target

    tfs = transforms.Compose([
        fake_segmentation,
        coco_utils.ConvertCocoPolysToMask(),
        presets.DetectionPresetTrain(data_augmentation='hflip') if train else presets.DetectionPresetEval(),
        # jeśli chcesz dodać swoje własne augmentacje, możesz zrobić to tutaj
    ])
    ds = coco_utils.CocoDetection(img_root, file_name, transforms=tfs)
    return ds


def load_trained_model(model, checkpoint_path):
    """Load trained model weights from checkpoint"""
    print(f"Loading trained model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


# Konfiguracja hiperparametrów
LR           = 0.001    # powinno być dobrze dla 1 GPU
WDECAY       = 0.0001
EPOCHS       = 25
VAL_FREQ     = 5        # walidacja i checkpointowanie co N epok
BATCH_SIZE   = 2        # dobierz pod możliwości sprzętowe
NUM_WORKERS  = 8        # j/w
NUM_CLASSES  = 14
DEVICE       = 'cuda:0'
DATASET_ROOT = 'dane/chess/'
WILD_DIR     = 'dane/wild/'
OUTPUT_DIR   = 'outputs/'
SEED         = 420


# Skipping parts of code
SKIP_TRAINING = True
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "checkpoint.pth")

RUN_ZADANIE_1  = False
RUN_ZADANIE_2a = False
RUN_ZADANIE_2b = True

# Zaczytanie datasetów
chess_train = get_dataset(os.path.join(DATASET_ROOT, 'train'), os.path.join(DATASET_ROOT, 'train/_annotations.coco.json'))
chess_val = get_dataset(os.path.join(DATASET_ROOT, 'valid'), os.path.join(DATASET_ROOT, 'valid/_annotations.coco.json'))

# Add generator for samplers
g = torch.Generator()
g.manual_seed(SEED)

# Update sampler initialization
train_sampler = torch.utils.data.RandomSampler(chess_train, generator=g)
train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, BATCH_SIZE, drop_last=True)

# Update DataLoader with worker init function
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)

train_loader = torch.utils.data.DataLoader(
    chess_train, 
    batch_sampler=train_batch_sampler, 
    num_workers=NUM_WORKERS, 
    collate_fn=utils.collate_fn,
    worker_init_fn=seed_worker,
    generator=g
)

val_sampler = torch.utils.data.SequentialSampler(chess_val)  # Sequential sampler doesn't need a generator
val_loader = torch.utils.data.DataLoader(
    chess_val, 
    batch_size=1, 
    sampler=val_sampler, 
    num_workers=NUM_WORKERS, 
    collate_fn=utils.collate_fn,
    worker_init_fn=seed_worker,
    generator=g
)
print("Data loaded")


# Skonstruowanie modelu; tworzymy w wersji dla 91 klas aby zainicjować wagi wstępnie na COCO...
model = M.fasterrcnn_resnet50_fpn(weights=M.FasterRCNN_ResNet50_FPN_Weights.COCO_V1, num_classes=91).to(DEVICE)
# ...po czym zastępujemy predyktor mniejszym, dostosowanym do naszego zbioru:
model.roi_heads.box_predictor = M.faster_rcnn.FastRCNNPredictor(in_channels=1024, num_classes=NUM_CLASSES).to(DEVICE)


print(model) # zwróć uwagę na strukturę Box Predictora (dlaczego tyle out_features?)


if SKIP_TRAINING and os.path.exists(CHECKPOINT_PATH):
    model = load_trained_model(model, CHECKPOINT_PATH)
    print("Loaded pre-trained model, skipping training")
else:
    print("Training one epoch")
    # Zanim przejdziemy do uczenia pełnego modelu, wykonamy krótkie wstępne uczenie losowo zainicjowanego predyktora:
    train_one_epoch(
        model=model,
        optimizer=torch.optim.AdamW(model.roi_heads.box_predictor.parameters(), lr=LR, weight_decay=WDECAY),
        data_loader=train_loader,
        device=DEVICE,
        epoch=0, print_freq=20, scaler=None
    )
    print("Training one epoch done")
    
    # Uczenie pełnego modelu
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=WDECAY
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    print("Training all epochs")
    start_time = time.perf_counter()
    for epoch in range(EPOCHS):
        train_one_epoch(model, optimizer, train_loader, DEVICE, epoch, 20, None)
        lr_scheduler.step()

        # eval and checkpoint every VAL_FREQ epochs
        if (epoch+1) % VAL_FREQ == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
            }
            utils.save_on_master(checkpoint, os.path.join(OUTPUT_DIR, f"model_{epoch}.pth"))
            utils.save_on_master(checkpoint, CHECKPOINT_PATH)
            evaluate(model, val_loader, device=DEVICE)

    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")

# Before inference, ensure model is in eval mode
model.eval()  
with torch.no_grad():
    # Inferencja na zadanym obrazie
    preprocess = M.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    img = read_image(os.path.join(DATASET_ROOT, 'test/IMG_0159_JPG.rf.1cf4f243b5072d63e492711720df35f7.jpg'))
    batch = [preprocess(img).to(DEVICE)]
    prediction = model(batch)[0]

# Rysowanie predykcji - wygodny gotowiec
box = draw_bounding_boxes(
  img,
  boxes=prediction['boxes'],
  labels=[chess_train.coco.cats[i.item()]['name'] for i in prediction['labels']],
  colors='red',
  width=4,
)

to_pil_image(box.detach()).save(os.path.join(OUTPUT_DIR, f"prediction.jpg"))


# ### Zadanie 1
# 
# Zbadaj wpływ parametrów inferencji **głowic `RoIHeads`**, progu prawdopodobieństwa (`score_thresh`) i progu NMS (`nms_thresh`), na działanie modelu. Wykorzystaj funkcję `evaluate` aby zmierzyć zmianę jakości predykcji, ale przebadaj też efekty wizualnie, wyświetlając predykcje dla kilku obrazów ze zbioru walidacyjnego i kilku spoza zbioru (folder `wild`). _W finalnej wersji pozostaw tylko wybrane interesujące przykłady._


def evaluate_model_params(model, val_loader, score_thresh, nms_thresh, device):
    
    original_score_thresh = model.roi_heads.score_thresh
    original_nms_thresh = model.roi_heads.nms_thresh
    
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    
    metrics = evaluate(model, val_loader, device=device)
    
    model.roi_heads.score_thresh = original_score_thresh
    model.roi_heads.nms_thresh = original_nms_thresh
    
    return metrics


def visualize_predictions(model, image_path, score_thresh, nms_thresh, dataset, save_path):
    model.roi_heads.score_thresh = score_thresh
    model.roi_heads.nms_thresh = nms_thresh
    
    img = read_image(image_path)
    preprocess = M.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    batch = [preprocess(img).to(DEVICE)]
    
    model.eval()
    with torch.no_grad():
        prediction = model(batch)[0]
    
    box = draw_bounding_boxes(
        img,
        boxes=prediction['boxes'],
        labels=[f"{dataset.coco.cats[i.item()]['name']}\n{s:.2f}" 
                for i, s in zip(prediction['labels'], prediction['scores'])],
        colors='red',
        width=4,
    )
    return box, len(prediction['boxes'])  # Return both the box and number of predictions


def evaluate_parameters(model, val_loader, score_thresholds, nms_thresholds, device):
    results = []
    total_combinations = len(score_thresholds) * len(nms_thresholds)
    
    with tqdm(total=total_combinations, desc="Evaluating parameters") as pbar:
        for score_t in score_thresholds:
            for nms_t in nms_thresholds:
                try:
                    metrics = evaluate_model_params(model, val_loader, score_t, nms_t, device)
                    results.append({
                        'score_thresh': score_t,
                        'nms_thresh': nms_t,
                        'ap': metrics.coco_eval['bbox'].stats[0]
                    })
                except Exception as e:
                    print(f"Error evaluating score_thresh={score_t}, nms_thresh={nms_t}: {str(e)}")
                pbar.update(1)
    
    return results


test_images = [
    os.path.join(DATASET_ROOT, 'valid/IMG_0293_JPG.rf.f29eab19f33f4c8ef04f9188d7ff1de7.jpg'),
    os.path.join(DATASET_ROOT, 'valid/IMG_0310_JPG.rf.6cf8e3d4550948ac9e5efafc66f1cdfd.jpg'),
    os.path.join(DATASET_ROOT, 'test/IMG_0169_JPG.rf.b1530b71278953ad465d06863135c71e.jpg'),
    os.path.join(WILD_DIR,     'chesscom_fide.jpeg'),
    os.path.join(WILD_DIR,     'shop.jpg'),
    os.path.join(WILD_DIR,     'moje.png'),
]


if RUN_ZADANIE_1:
    print("\n\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
    print("Zadanie 1\n")

    plot_dir = Path(OUTPUT_DIR) / 'zad1_parameter_analysis'
    plot_dir.mkdir(exist_ok=True)

    # Parameters to test
    score_thresholds = [0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.75, 0.9]
    nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = evaluate_parameters(model, val_loader, score_thresholds, nms_thresholds, DEVICE)

    create_heatmap(results, score_thresholds, nms_thresholds, plot_dir)

    # Add precision-recall curve analysis
    pr_score_thresholds = [0.1, 0.5, 0.75]
    pr_nms_thresholds   = [0.1, 0.4, 0.7]
    plot_precision_recall_curves(model, val_loader, pr_score_thresholds, pr_nms_thresholds, DEVICE, plot_dir)

    vis_score_thresholds = score_thresholds
    vis_nms_thresholds = nms_thresholds

    with tqdm(total=len(test_images), desc="Visualizing predictions") as pbar:
        for img_path in test_images:
            print(f"Visualizing predictions for {Path(img_path).name}")
            fig, axes = plt.subplots(len(vis_score_thresholds), len(vis_nms_thresholds), figsize=(40, 40))
            fig.suptitle(f'Predictions for {Path(img_path).name}', fontsize=16, y=0.95)
            
            for i, score_t in enumerate(vis_score_thresholds):
                for j, nms_t in enumerate(vis_nms_thresholds):
                    box, num_preds = visualize_predictions(model, img_path, score_t, nms_t, chess_train, plot_dir)
                    axes[i, j].imshow(box.permute(1, 2, 0))
                    axes[i, j].axis('off')
                    axes[i, j].set_title(f'S_th={score_t}, NMS_th={nms_t}, pred={num_preds}', fontsize=12)
            
            plt.savefig(plot_dir / f'parameter_grid_{Path(img_path).stem}.png', 
                    bbox_inches='tight', dpi=300)
            plt.close()
            pbar.update(1)


SCORE_THRESH = 0.75
NMS_THRESH   = 0.4


# ### Zadanie 2a
# Zwizualizuj propozycje rejonów wygenerowane przez RPN i porównaj z ostateczną predykcją.
# 
# W tym celu konieczne będzie manualne wykonanie fragmentu metody `GeneralizedRCNN::forward` (patrz: [kod](https://github.com/pytorch/vision/blob/6279faa88a3fe7de49bf58284d31e3941b768522/torchvision/models/detection/generalized_rcnn.py#L46), link do wersji najnowszej na grudzień 2024).
# Wszystkie fragmenty związane z uczeniem możesz rzecz jasna pominąć; chodzi o wyciągnięcie obiektu `proposals`.
# Nie zapomnij o wykonaniu powrotnej transformacji! (Po co?)


def get_rpn_proposals(model, image_path):

    img = read_image(image_path)
    preprocess = M.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    processed_img = preprocess(img)
    
    # Create batch of one image
    images = [processed_img.to(DEVICE)]
    
    model.eval()
    with torch.no_grad():

        transformed_images, _ = model.transform(images, None)
        
        # features from backbone
        features = model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        # RPN proposals
        proposals, _ = model.rpn(transformed_images, features, None)
        
        # final predictions
        detections, _ = model.roi_heads(features, proposals, transformed_images.image_sizes, None)
        detections = model.transform.postprocess(detections, transformed_images.image_sizes, [img.shape[1:]])
        
        # Transform proposals back to original image space
        proposal_boxes = proposals[0]
        orig_proposal_boxes = model.transform.postprocess([{'boxes': proposal_boxes}], 
                                                        transformed_images.image_sizes, 
                                                        [img.shape[1:]])[0]['boxes']
        
    return img, orig_proposal_boxes, detections[0]



if RUN_ZADANIE_2a:
    print("\n\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
    print("Zadanie 2a\n")

    model.roi_heads.score_thresh = SCORE_THRESH
    model.roi_heads.nms_thresh = NMS_THRESH

    plot_dir = Path(OUTPUT_DIR) / 'zad2a_rpn_analysis'
    plot_dir.mkdir(exist_ok=True)

    for img_path in test_images:
        print(f"Processing {Path(img_path).name}")
        img, proposals, predictions = get_rpn_proposals(model, img_path)
        
        save_path = plot_dir / f'rpn_analysis_{Path(img_path).stem}.png'
        visualize_proposals_and_predictions(img, proposals, predictions, chess_train, save_path)
        
        print(f"Number of RPN proposals: {len(proposals)}")
        print(f"Number of final predictions: {len(predictions['boxes'])}")
        print()


# ### Zadanie 2b
# 
# Zbadaj wpływ progu NMS _na etapie propozycji_ na jakość predykcji oraz czas ich uzyskania.
# Jak w poprzednich zadaniach, postaraj się nie ograniczyć tylko do pokazania metryk, ale pokaż wizualizacje (propozycji i predykcji) dla **wybranych** przykładów.


def get_rpn_proposals_with_nms(model, image_path, rpn_nms_thresh):
    img = read_image(image_path)
    preprocess = M.FasterRCNN_ResNet50_FPN_Weights.COCO_V1.transforms()
    processed_img = preprocess(img)
    images = [processed_img.to(DEVICE)]
    
    model.eval()
    times = {}
    
    start_time = time.perf_counter()
    with torch.no_grad():
        # Store original NMS threshold
        original_nms_thresh = model.rpn.nms_thresh
        model.rpn.nms_thresh = rpn_nms_thresh
        
        # Forward pass through model components
        transformed_images, _ = model.transform(images, None)
        
        # Time backbone separately
        t0 = time.perf_counter()
        features = model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        t1 = time.perf_counter()
        
        proposals, _ = model.rpn(transformed_images, features, None)
        t2 = time.perf_counter()
        
        detections, _ = model.roi_heads(features, proposals, transformed_images.image_sizes, None)
        t3 = time.perf_counter()
        
        detections = model.transform.postprocess(detections, transformed_images.image_sizes, [img.shape[1:]])
        proposal_boxes = proposals[0]
        
        orig_proposal_boxes = model.transform.postprocess(
            [{'boxes': proposal_boxes}], 
            transformed_images.image_sizes, 
            [img.shape[1:]]
        )[0]['boxes']
        
        model.rpn.nms_thresh = original_nms_thresh
    
    end_time = time.perf_counter()
    times = {
        'backbone_time': t1 - t0,
        'rpn_time': t2 - t1,
        'roi_time': t3 - t2,
        'total_time': end_time - start_time
    }
    
    # print(f'Original RPN NMS threshold: {original_nms_thresh}')

    return img, orig_proposal_boxes, detections[0], times


def evaluate_rpn_nms_threshold(model, val_loader, rpn_nms_thresh, device):

    original_nms_thresh = model.rpn.nms_thresh
    model.rpn.nms_thresh = rpn_nms_thresh
    metrics = evaluate(model, val_loader, device=device)
    model.rpn.nms_thresh = original_nms_thresh
    return metrics


if RUN_ZADANIE_2b:
    print("\n\n# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
    print("Zadanie 2b\n")
    
    plot_dir = Path(OUTPUT_DIR) / 'zad2b_rpn_nms_analysis'
    plot_dir.mkdir(exist_ok=True)
    
    # Test different RPN NMS thresholds
    rpn_nms_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    
    model.roi_heads.score_thresh = SCORE_THRESH
    model.roi_heads.nms_thresh = NMS_THRESH
    
    results = []
    for nms_t in rpn_nms_thresholds:
        print(f"\nEvaluating RPN NMS threshold = {nms_t}")
        metrics = evaluate_rpn_nms_threshold(model, val_loader, nms_t, DEVICE)
        results.append({
            'rpn_nms_thresh': nms_t,
            'ap': metrics.coco_eval['bbox'].stats[0]
        })
    
    visualize_rpn_nms_analysis(results, plot_dir)
    rpn_nms_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    timing_results = []
    
    for img_path in test_images:
        print(f"\nProcessing {Path(img_path).name}")
        
        # Create figure with subplots for each threshold
        fig = plt.figure(figsize=(20, 10*len(rpn_nms_thresholds)))
        
        # Create a grid of subplots
        gs = plt.GridSpec(len(rpn_nms_thresholds), 2, figure=fig)
        
        for i, nms_t in enumerate(rpn_nms_thresholds):
            img, proposals, predictions, times = get_rpn_proposals_with_nms(model, img_path, nms_t)
            
            timing_results.append({
                'image': Path(img_path).name,
                'rpn_nms_thresh': nms_t,
                **times
            })
            
            # Create side-by-side visualization
            proposals_vis = draw_bounding_boxes(img.clone(), boxes=proposals, colors='green', width=1)
            predictions_vis = draw_bounding_boxes(
                img.clone(),
                boxes=predictions['boxes'],
                labels=[f"{chess_train.coco.cats[i.item()]['name']}\n{s:.2f}" 
                       for i, s in zip(predictions['labels'], predictions['scores'])],
                colors='red',
                width=2
            )
            
            # Plot both visualizations side by side using GridSpec
            ax1 = fig.add_subplot(gs[i, 0])
            ax2 = fig.add_subplot(gs[i, 1])
            
            ax1.imshow(proposals_vis.permute(1, 2, 0))
            ax1.set_title(f'RPN Proposals (thresh={nms_t})\n{len(proposals)} proposals\n' + 
                          f'Backbone time: {times["backbone_time"]:.3f}s\n' +
                          f'RPN time: {times["rpn_time"]:.3f}s', fontsize=12)
            ax1.axis('off')
            
            ax2.imshow(predictions_vis.permute(1, 2, 0))
            ax2.set_title(f'Final Predictions\n{len(predictions["boxes"])} detections\nROI time: {times["roi_time"]:.3f}s', fontsize=12)
            ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig(plot_dir / f'rpn_nms_analysis_{Path(img_path).stem}.png', bbox_inches='tight', dpi=300)
        plt.close()
    
    visualize_timing_analysis(timing_results, plot_dir, rpn_nms_thresholds)

print("Done :>")