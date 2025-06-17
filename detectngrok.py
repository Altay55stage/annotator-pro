import os
import sys
import json
import base64
from io import BytesIO
import traceback
import shutil
import random
import yaml
from typing import Optional, List, Dict, Any, Tuple
import uuid
from pathlib import Path
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager
import zipfile
import csv
import logging

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import torch
import requests

# --- Configuration du Logging ---
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] SERVER: %(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

# --- Configuration Globale ---
HOME_DIR = Path("/home/acevik").resolve()
MODELS_DIR = HOME_DIR
TEMP_UPLOAD_DIR = HOME_DIR / "temp_uploads_annotator"
FINETUNE_PROJECTS_DIR = HOME_DIR / "finetune_projects_annotator"
FINETUNED_MODELS_OUTPUT_DIR = HOME_DIR / "finetuned_model_runs_annotator"
TASKS_DB_FILE = HOME_DIR / "annotator_tasks_db.json"

for p_subdir in [TEMP_UPLOAD_DIR, FINETUNE_PROJECTS_DIR, FINETUNED_MODELS_OUTPUT_DIR]:
    p_subdir.mkdir(parents=True, exist_ok=True)

# --- Configuration des valeurs par défaut ---
DEFAULT_CONFIDENCE = 0.3
DEFAULT_IMG_SIZE_INT = 640
DEFAULT_NMS_IOU_FLOAT = 0.45
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEFAULT_FINETUNE_EPOCHS = 25
DEFAULT_FINETUNE_BATCH_SIZE = 4
DEFAULT_FINETUNE_LR0 = 0.01
DEFAULT_FINETUNE_OPTIMIZER = 'auto'

# --- Configuration API LLM Externe ---
EXTERNAL_NGROK_LLM_BASE_URL = "https://80f2-193-55-66-240.ngrok-free.app"
EXTERNAL_LLM_API_ENDPOINT = "/api/llm/describe"
EXTERNAL_LLM_FULL_URL = EXTERNAL_NGROK_LLM_BASE_URL + EXTERNAL_LLM_API_ENDPOINT

SAHI_AVAILABLE = False
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
    logger.info("Librairie SAHI détectée et importée.")
except ImportError:
    logger.warning("Librairie SAHI non installée.")
except Exception as e:
    logger.error(f"Erreur import SAHI: {e}")

# --- Modèles Pydantic ---
class ModelInfo(BaseModel):
    name: str
    path: str

class LoadModelRequest(BaseModel):
    model_name: str

class BoundingBox(BaseModel):
    x1: float; y1: float; x2: float; y2: float

class AnnotatedBoundingBox(BoundingBox):
    label: str

class Detection(BaseModel):
    box: BoundingBox
    label: str
    confidence: float

class DetectionResponse(BaseModel):
    detections: List[Detection]
    image_id: str
    original_filename: str
    message: str

class LLMDescribeRequest(BaseModel):
    image_base64: str
    prompt: str = "Décris cette image en une ou deux phrases concises."

class LLMDescribeResponse(BaseModel):
    description: str

class ProjectCreateRequest(BaseModel):
    name: str = Field(..., example="road_defects_v1", min_length=3, max_length=50)
    description: Optional[str] = Field(None, example="Dataset pour la détection de défauts routiers", max_length=250)

class ProjectDatasetStats(BaseModel):
    class_distribution: Dict[str, int] = {}
    avg_boxes_per_image: float = 0.0
    image_size_distribution: Dict[str, int] = {}

class ProjectInfo(ProjectCreateRequest):
    project_id: str
    created_at: datetime
    image_count: int = 0
    annotation_count: int = 0
    classes: List[str] = []
    class_mapping: Dict[str, int] = {}
    dataset_yaml_path: Optional[str] = None
    train_image_count: int = 0
    val_image_count: int = 0
    test_image_count: int = 0
    is_split: bool = False
    stats: Optional[ProjectDatasetStats] = None

class AddImageToProjectRequest(BaseModel):
    temp_image_id: str
    original_filename: str
    annotations: List[AnnotatedBoundingBox]
    target_set: str = Field("train", pattern="^(train|val|test)$")

class UploadDatasetZipInfo(BaseModel):
    message: str
    project_id: str
    images_added: int
    labels_found: int
    new_classes_added: List[str]
    updated_project_info: ProjectInfo

class SplitDatasetRequest(BaseModel):
    train_ratio: float = Field(0.7, ge=0, le=1)
    val_ratio: float = Field(0.2, ge=0, le=1)
    test_ratio: float = Field(0.1, ge=0, le=1)
    shuffle: bool = True

class FineTuneAugmentationConfig(BaseModel):
    hsv_h: float = Field(0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(0.4, ge=0.0, le=1.0)
    degrees: float = Field(0.0, ge=0.0, le=360.0)
    translate: float = Field(0.1, ge=0.0, le=1.0)
    scale: float = Field(0.5, ge=0.0, le=1.0)
    shear: float = Field(0.0, ge=0.0, le=45.0)
    perspective: float = Field(0.0, ge=0.0, le=0.001)
    flipud: float = Field(0.0, ge=0.0, le=1.0)
    fliplr: float = Field(0.5, ge=0.0, le=1.0)
    mosaic: float = Field(1.0, ge=0.0, le=1.0)
    mixup: float = Field(0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(0.0, ge=0.0, le=1.0)

class FineTuneRunConfig(BaseModel):
    epochs: int = DEFAULT_FINETUNE_EPOCHS
    batch: int = DEFAULT_FINETUNE_BATCH_SIZE
    imgsz: int = DEFAULT_IMG_SIZE_INT
    optimizer: str = DEFAULT_FINETUNE_OPTIMIZER
    lr0: float = DEFAULT_FINETUNE_LR0
    lrf: float = Field(0.01, ge=0.0, le=1.0)
    device: str = DEFAULT_DEVICE
    patience: int = Field(50, ge=0)
    augmentations: FineTuneAugmentationConfig = Field(default_factory=FineTuneAugmentationConfig)

class FineTuneRequest(BaseModel):
    project_id: str
    base_model_path: str
    run_config: FineTuneRunConfig
    run_name: Optional[str] = None
    save_period: int = Field(-1, ge=-1)

class TaskStatus(BaseModel):
    task_id: str
    task_type: str
    status: str
    message: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    result: Optional[Dict[str, Any]] = None
    progress: Optional[float] = None
    sub_tasks_status: Optional[List[Dict]] = None

class TaskStatusResponse(TaskStatus):
    pass

class RunArtifact(BaseModel):
    name: str
    path_on_server_relative_to_run_dir: str
    type: str

class FineTunedModelInfo(BaseModel):
    run_name: str
    model_name_pt: str
    model_path_abs: str
    model_path_relative_to_home: str
    project_id_source: str
    base_model_used: str
    training_date: datetime
    run_config_used: Dict[str, Any]
    final_metrics: Optional[Dict[str, float]] = None
    dataset_yaml_used_path_abs: Optional[str] = None
    artifacts: List[RunArtifact] = []
    output_dir_abs: str
    output_dir_relative_to_home: str
    best_epoch: Optional[int] = None
    notes: Optional[str] = None

class ExportFormat(BaseModel):
    format: str = Field("onnx", pattern="^(onnx|torchscript|engine|tflite|pb|coreml)$")
    imgsz: Optional[int] = None
    half: bool = False

class HyperparameterTuneRequest(BaseModel):
    project_id: str
    base_model_path: str
    base_run_config: FineTuneRunConfig
    num_trials: int = Field(10, ge=1, le=100)

# --- Initialisation FastAPI ---
app = FastAPI(title="YOLO Annotator Pro - Advanced ML Suite", version="1.0.3")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- État Global du Serveur ---
current_model_instance: Optional[YOLO] = None
current_model_path_loaded: Optional[Path] = None
current_model_type_loaded: Optional[str] = None

# --- Gestion des tâches ---
def load_tasks_db() -> Dict[str, TaskStatus]:
    if TASKS_DB_FILE.exists():
        try:
            with open(TASKS_DB_FILE, 'r') as f:
                tasks_raw = json.load(f)
            parsed_tasks = {}
            for task_id, data in tasks_raw.items():
                for date_key in ["created_at", "updated_at"]:
                    if date_key in data and isinstance(data[date_key], str):
                        date_str = data[date_key]
                        if date_str.endswith('Z'):
                            date_str = date_str[:-1] + '+00:00'
                        try:
                            data[date_key] = datetime.fromisoformat(date_str)
                        except ValueError as e_iso:
                            logger.error(f"Impossible de parser la date '{data[date_key]}' pour la tâche {task_id}, clé {date_key}: {e_iso}. Utilisation de la date actuelle.")
                            data[date_key] = datetime.now(timezone.utc)
                try:
                    parsed_tasks[task_id] = TaskStatus(**data)
                except Exception as e_pydantic:
                    logger.error(f"Erreur de validation Pydantic pour la tâche {task_id}: {e_pydantic}. Données: {data}")
            return parsed_tasks
        except json.JSONDecodeError as e_json:
            logger.error(f"Erreur de décodage JSON pour tasks_db.json: {e_json}. Le fichier est peut-être corrompu. Renommage en .corrupted.")
            try:
                corrupted_path = TASKS_DB_FILE.with_suffix(f'.corrupted_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
                TASKS_DB_FILE.rename(corrupted_path)
            except Exception as e_rename:
                logger.error(f"Impossible de renommer le fichier tasks_db.json corrompu: {e_rename}")
            return {}
        except Exception as e:
            logger.error(f"Erreur inattendue lors du chargement de tasks_db.json: {e}", exc_info=True)
    return {}

def save_tasks_db(tasks: Dict[str, TaskStatus]):
    try:
        with open(TASKS_DB_FILE, 'w') as f:
            json.dump({task_id: task.model_dump(mode='json') for task_id, task in tasks.items()}, f, indent=4)
    except Exception as e:
        logger.error(f"Erreur sauvegarde tasks_db.json: {e}", exc_info=True)

# --- Fonctions Utilitaires ---
def get_project_dir(project_id: str) -> Path:
    return FINETUNE_PROJECTS_DIR / project_id

def get_project_images_dir(project_id: str, subset: Optional[str] = None) -> Path:
    base = get_project_dir(project_id) / "images"
    return base / subset if subset else base

def get_project_labels_dir(project_id: str, subset: Optional[str] = None) -> Path:
    base = get_project_dir(project_id) / "labels"
    return base / subset if subset else base

def get_project_metadata_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "_project_metadata.json"

def get_project_dataset_yaml_path(project_id: str) -> Path:
    return get_project_dir(project_id) / "dataset.yaml"

def get_run_dir(run_name: str) -> Path:
    return FINETUNED_MODELS_OUTPUT_DIR / run_name

def get_run_metadata_path(run_name: str) -> Path:
    return get_run_dir(run_name) / "_run_metadata.json"

def load_project_metadata(project_id: str) -> Optional[ProjectInfo]:
    meta_path = get_project_metadata_path(project_id)
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            try:
                data = json.load(f)
                if isinstance(data.get("created_at"), str):
                    created_at_str = data["created_at"]
                    if created_at_str.endswith('Z'):
                        created_at_str = created_at_str[:-1] + '+00:00'
                    data["created_at"] = datetime.fromisoformat(created_at_str)
                if "stats" in data and data["stats"] is not None and not isinstance(data["stats"], dict):
                    data["stats"] = None
                return ProjectInfo(**data)
            except (json.JSONDecodeError, TypeError, ValueError) as e:
                logger.error(f"Erreur décodage/validation métadonnées projet {project_id}: {e}", exc_info=True)
                return None
    return None

def save_project_metadata(project_id: str, metadata_model: ProjectInfo):
    meta_path = get_project_metadata_path(project_id)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(metadata_model.model_dump(mode='json'), f, indent=4)

def _update_project_image_counts_and_stats(project_id: str, metadata: ProjectInfo) -> ProjectInfo:
    counts = {"train": 0, "val": 0, "test": 0}
    is_split_detected = False
    all_image_paths_in_sets = []

    for s in ["train", "val", "test"]:
        img_s_dir = get_project_images_dir(project_id, s)
        if img_s_dir.exists():
            subset_images = [f for f in img_s_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
            counts[s] = len(subset_images)
            all_image_paths_in_sets.extend(subset_images)
            if counts[s] > 0:
                is_split_detected = True
    
    metadata.train_image_count = counts["train"]
    metadata.val_image_count = counts["val"]
    metadata.test_image_count = counts["test"]
    
    root_images_dir_content = []
    if not is_split_detected:
        root_images_dir = get_project_images_dir(project_id)
        if root_images_dir.exists():
            root_images_dir_content = [f for f in root_images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] and not f.name.startswith('.')]
            metadata.image_count = len(root_images_dir_content)
    else:
        metadata.image_count = len(all_image_paths_in_sets)

    metadata.is_split = is_split_detected

    class_distribution = {}
    total_boxes = 0
    image_size_distribution = {}
    
    image_source_dirs_for_stats = []
    label_source_dirs_for_stats = []

    if is_split_detected:
        for s in ["train", "val", "test"]:
            if counts[s] > 0:
                image_source_dirs_for_stats.append(get_project_images_dir(project_id, s))
                label_source_dirs_for_stats.append(get_project_labels_dir(project_id, s))
    elif root_images_dir_content:
        image_source_dirs_for_stats.append(get_project_images_dir(project_id))
        label_source_dirs_for_stats.append(get_project_labels_dir(project_id))

    if metadata.class_mapping and (is_split_detected or root_images_dir_content) :
        id_to_name_map = {v: k for k, v in metadata.class_mapping.items()}
        images_processed_for_size = set()

        for img_dir, lbl_dir in zip(image_source_dirs_for_stats, label_source_dirs_for_stats):
            if not lbl_dir.exists(): continue
            for label_file in lbl_dir.glob("*.txt"):
                image_file_stem = label_file.stem
                if image_file_stem not in images_processed_for_size:
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        potential_img_path = img_dir / (image_file_stem + ext)
                        if potential_img_path.exists():
                            try:
                                img = Image.open(potential_img_path)
                                size_str = f"{img.width}x{img.height}"
                                image_size_distribution[size_str] = image_size_distribution.get(size_str, 0) + 1
                                images_processed_for_size.add(image_file_stem)
                            except Exception: pass
                            break

                with open(label_file, 'r') as lf:
                    lines = lf.readlines()
                    total_boxes += len(lines)
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            try:
                                class_id = int(parts[0])
                                class_name = id_to_name_map.get(class_id, f"unknown_id_{class_id}")
                                class_distribution[class_name] = class_distribution.get(class_name, 0) + 1
                            except ValueError:
                                logger.warning(f"Ligne label invalide dans {label_file}: {line.strip()}")
        
        metadata.annotation_count = total_boxes
        avg_boxes = total_boxes / metadata.image_count if metadata.image_count > 0 else 0.0
        metadata.stats = ProjectDatasetStats(
            class_distribution=class_distribution,
            avg_boxes_per_image=round(avg_boxes, 2),
            image_size_distribution=image_size_distribution
        )
    else:
        metadata.annotation_count = 0
        metadata.stats = ProjectDatasetStats()
    return metadata

def update_project_dataset_yaml(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata:
        logger.error(f"Métadonnées projet {project_id} non trouvées pour màj YAML.")
        return

    project_dir_abs = get_project_dir(project_id).resolve()
    metadata = _update_project_image_counts_and_stats(project_id, metadata)

    class_mapping = metadata.class_mapping
    class_names_ordered = []
    if class_mapping and isinstance(class_mapping, dict):
        sorted_class_items = sorted(class_mapping.items(), key=lambda item: item[1])
        expected_id = 0
        for name, c_id in sorted_class_items:
            if c_id == expected_id:
                class_names_ordered.append(name)
                expected_id += 1
            else:
                logger.error(f"Rupture ID classes pour {project_id}. Attendu: {expected_id}, Obtenu: {c_id} pour '{name}'. YAML sera invalide.")
                class_names_ordered = []
                break
    
    if not class_names_ordered and metadata.image_count > 0:
            logger.warning(f"Projet {project_id} a des images mais pas de classes définies. YAML aura nc=0.")

    yaml_content = {
        "path": str(project_dir_abs),
        "nc": len(class_names_ordered),
        "names": class_names_ordered
    }

    if metadata.image_count > 0:
        if metadata.is_split:
            # Always include train key
            yaml_content["train"] = str(Path("images") / "train")
            
            # FIX: If val set is empty, point to train set to avoid crash.
            if metadata.val_image_count > 0:
                yaml_content["val"] = str(Path("images") / "val")
            else:
                logger.warning(f"Le set de validation pour le projet {project_id} est vide. Utilisation du set d'entraînement pour la validation afin d'éviter un crash.")
                yaml_content["val"] = str(Path("images") / "train")
            
            # Test set is optional
            if metadata.test_image_count > 0:
                yaml_content["test"] = str(Path("images") / "test")
        else:
            # For non-split datasets, point both to the root images directory
            yaml_content["train"] = "images"
            yaml_content["val"] = "images"

    yaml_path = get_project_dataset_yaml_path(project_id)
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)
        logger.info(f"dataset.yaml projet {project_id} màj.")
        metadata.dataset_yaml_path = str(yaml_path.resolve())
    except Exception as e:
        logger.error(f"Erreur écriture dataset.yaml pour {project_id}: {e}", exc_info=True)
        metadata.dataset_yaml_path = None
    
    save_project_metadata(project_id, metadata)


def _create_task_entry(task_type: str, message: str = "Tâche initialisée") -> TaskStatus:
    task_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)
    return TaskStatus(task_id=task_id, task_type=task_type, status="pending", message=message, created_at=now, updated_at=now)

def _update_task_status(task_id: str, status: str, message: str, result: Optional[Dict] = None, progress: Optional[float] = None):
    tasks = load_tasks_db()
    task = tasks.get(task_id)
    if task:
        task.status = status
        task.message = message
        task.updated_at = datetime.now(timezone.utc)
        if result is not None: task.result = result
        if progress is not None: task.progress = max(0.0, min(1.0, progress))
        save_tasks_db(tasks)
    else:
        logger.warning(f"Tentative de mise à jour d'une tâche inexistante: {task_id}")

# --- Lifespan & Endpoints ---
@asynccontextmanager
async def lifespan(app_fastapi: FastAPI):
    logger.info("Serveur YOLO Annotator Pro (Advanced ML Suite) démarré.")
    yield
    logger.info("Serveur YOLO Annotator Pro (Advanced ML Suite) arrêté.")

app.router.lifespan_context = lifespan

@app.get("/status", tags=["General"])
async def get_server_status():
    return {
        "status": "running", "timestamp": datetime.now(timezone.utc),
        "torch_version": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "sahi_available": SAHI_AVAILABLE, "home_dir": str(HOME_DIR)
    }

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models_endpoint():
    models_list = []
    if not MODELS_DIR.is_dir():
        raise HTTPException(status_code=500, detail=f"Dossier des modèles ({MODELS_DIR}) non trouvé.")
    try:
        for item_name in os.listdir(MODELS_DIR):
            item_path = MODELS_DIR / item_name
            if item_path.is_file() and item_name.lower().endswith((".pt", ".pth")):
                models_list.append(ModelInfo(name=item_name, path=str(item_path.resolve())))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors du listage des modèles: {str(e)}")
    return models_list

@app.post("/load_model", tags=["Models"])
async def load_model_endpoint(request: LoadModelRequest):
    global current_model_instance, current_model_path_loaded, current_model_type_loaded
    model_name_or_path = request.model_name
    prospective_path_abs = Path(model_name_or_path)
    model_to_load_path = None

    if prospective_path_abs.is_absolute() and prospective_path_abs.is_file():
        model_to_load_path = prospective_path_abs
    else:
        path_rel_to_home = (HOME_DIR / model_name_or_path).resolve()
        if path_rel_to_home.is_file():
            model_to_load_path = path_rel_to_home
        else:
             simple_name_path = (HOME_DIR / Path(model_name_or_path).name).resolve()
             if simple_name_path.is_file():
                 model_to_load_path = simple_name_path
    
    if not model_to_load_path:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name_or_path}' non trouvé.")

    current_model_instance, current_model_path_loaded, current_model_type_loaded = None, None, None
    try:
        loaded_yolo_obj = YOLO(str(model_to_load_path))
        _ = loaded_yolo_obj.predict(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False, device='cpu') 
        current_model_instance = loaded_yolo_obj
        current_model_path_loaded = model_to_load_path
        current_model_type_loaded = "ultralytics"
        return {"message": f"Modèle '{model_to_load_path.name}' (Ultralytics) chargé avec succès."}
    except Exception as e_ultralytics:
        raise HTTPException(status_code=500, detail=f"Échec du chargement du modèle '{model_to_load_path.name}': {str(e_ultralytics)}.")

@app.post("/predict", response_model=DetectionResponse, tags=["Inference"])
async def predict_image(
    file: UploadFile = File(...),
    confidence: float = Query(DEFAULT_CONFIDENCE, ge=0.01, le=1.0),
    img_size: int = Query(DEFAULT_IMG_SIZE_INT, ge=32),
    nms_iou: float = Query(DEFAULT_NMS_IOU_FLOAT, ge=0.01, le=1.0),
    use_sahi: bool = Query(False),
    sahi_slice_height: int = Query(512, ge=128),
    sahi_slice_width: int = Query(512, ge=128),
    sahi_overlap_height_ratio: float = Query(0.2, ge=0.0, le=0.99),
    sahi_overlap_width_ratio: float = Query(0.2, ge=0.0, le=0.99),
    device: Optional[str] = Query(None)
):
    if not current_model_instance:
        raise HTTPException(status_code=400, detail="Aucun modèle n'est actuellement chargé.")

    original_filename = file.filename or f"image_{uuid.uuid4().hex[:6]}"
    file_extension = Path(original_filename).suffix.lower() or ".png"
    if file_extension not in [".png", ".jpg", ".jpeg", ".bmp", ".webp"]:
        raise HTTPException(status_code=400, detail=f"Type de fichier image non supporté: {file_extension}.")

    temp_image_id = f"{uuid.uuid4().hex}{file_extension}"
    temp_image_path = TEMP_UPLOAD_DIR / temp_image_id
    
    try:
        contents = await file.read()
        with open(temp_image_path, "wb") as f: f.write(contents)
        pil_image = Image.open(BytesIO(contents)).convert("RGB")
        image_np = np.array(pil_image) 
    except Exception as e:
        if temp_image_path.exists(): temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Erreur lecture/sauvegarde image: {str(e)}")

    actual_device_to_use = device if device else (current_model_instance.device if hasattr(current_model_instance, 'device') and current_model_instance.device else DEFAULT_DEVICE)
    if actual_device_to_use == 'cuda' and not torch.cuda.is_available(): actual_device_to_use = 'cpu'
    
    response_detections: List[Detection] = []
    try:
        if current_model_type_loaded == "ultralytics":
            if use_sahi and SAHI_AVAILABLE:
                sahi_model = AutoDetectionModel.from_pretrained(
                    model_type='yolov8', model_path=str(current_model_path_loaded),
                    confidence_threshold=confidence, device=actual_device_to_use,
                )
                sahi_results = get_sliced_prediction(
                    image_np, sahi_model,
                    slice_height=sahi_slice_height, slice_width=sahi_slice_width,
                    overlap_height_ratio=sahi_overlap_height_ratio, overlap_width_ratio=sahi_overlap_width_ratio
                )
                names_map = current_model_instance.names
                for pred in sahi_results.object_prediction_list:
                    box = pred.bbox
                    response_detections.append(Detection(
                        box=BoundingBox(x1=box.minx, y1=box.miny, x2=box.maxx, y2=box.maxy),
                        label=names_map.get(pred.category.id, f"class_{pred.category.id}"),
                        confidence=pred.score.value
                    ))
            else:
                yolo_results_list = current_model_instance.predict(
                    source=pil_image, conf=confidence, iou=nms_iou, imgsz=img_size,
                    device=actual_device_to_use, verbose=False
                )
                if yolo_results_list and yolo_results_list[0].boxes:
                    res = yolo_results_list[0]
                    names_map = res.names
                    for i in range(len(res.boxes)):
                        box_tensor = res.boxes.xyxy[i].cpu().numpy().tolist()
                        class_id = int(res.boxes.cls[i].cpu().item())
                        conf_score = float(res.boxes.conf[i].cpu().item())
                        label_name = names_map.get(class_id, f"class_{class_id}")
                        response_detections.append(Detection(
                            box=BoundingBox(x1=box_tensor[0], y1=box_tensor[1], x2=box_tensor[2], y2=box_tensor[3]),
                            label=label_name, confidence=conf_score
                        ))
        else:
            raise NotImplementedError(f"Type de modèle '{current_model_type_loaded}' non supporté.")

        return DetectionResponse(
            detections=response_detections, image_id=temp_image_id,
            original_filename=original_filename, message="Détection terminée."
        )
    except Exception as e:
        if temp_image_path.exists(): temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur détection: {str(e)}")

@app.post("/finetune_projects/{project_id}/upload_dataset_zip", response_model=UploadDatasetZipInfo, tags=["Projects Data"])
async def upload_dataset_zip_to_project(
    project_id: str,
    zip_file: UploadFile = File(...),
    default_target_set: str = Form("train"),
):
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")

    if default_target_set not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="`default_target_set` doit être 'train', 'val', ou 'test'.")

    temp_zip_path = TEMP_UPLOAD_DIR / f"{project_id}_{uuid.uuid4().hex}.zip"
    temp_extract_path = TEMP_UPLOAD_DIR / f"extracted_{project_id}_{uuid.uuid4().hex}"

    try:
        with open(temp_zip_path, "wb") as f_zip:
            shutil.copyfileobj(zip_file.file, f_zip)
        
        temp_extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        
        images_added_count = 0
        labels_found_count = 0
        new_classes_this_upload = set()

        for root, _, files in os.walk(temp_extract_path):
            relative_root = Path(root).relative_to(temp_extract_path)
            current_target_set = default_target_set
            path_parts = list(relative_root.parts)

            if path_parts:
                if path_parts[0].lower() in ["images", "labels"] and len(path_parts) > 1 and path_parts[1].lower() in ["train", "val", "test"]:
                    current_target_set = path_parts[1].lower()
                elif path_parts[0].lower() in ["train", "val", "test"]:
                     current_target_set = path_parts[0].lower()

            for filename in files:
                file_path_in_zip = Path(root) / filename
                
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    original_img_name = Path(filename)
                    img_fname_stem_in_proj = f"{original_img_name.stem}_{uuid.uuid4().hex[:8]}"
                    img_fname_in_proj = f"{img_fname_stem_in_proj}{original_img_name.suffix}"
                    
                    dest_img_dir = get_project_images_dir(project_id, current_target_set)
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                    dest_img_path = dest_img_dir / img_fname_in_proj
                    
                    shutil.copy(file_path_in_zip, dest_img_path)
                    images_added_count += 1
                    
                    original_label_filename = f"{original_img_name.stem}.txt"
                    label_path_in_zip_alongside = file_path_in_zip.with_name(original_label_filename)
                    
                    label_path_in_zip_parallel = None
                    if "images" in path_parts:
                        parallel_label_path_parts = [p if p != "images" else "labels" for p in path_parts]
                        label_path_in_zip_parallel = temp_extract_path / Path(*parallel_label_path_parts) / original_label_filename
                    elif len(path_parts) == 1 and path_parts[0] in ["train", "val", "test"]:
                        label_path_in_zip_parallel = temp_extract_path / "labels" / path_parts[0] / original_label_filename

                    found_label_path_in_zip = None
                    if label_path_in_zip_alongside.exists():
                        found_label_path_in_zip = label_path_in_zip_alongside
                    elif label_path_in_zip_parallel and label_path_in_zip_parallel.exists():
                        found_label_path_in_zip = label_path_in_zip_parallel
                    
                    if found_label_path_in_zip:
                        dest_lbl_dir = get_project_labels_dir(project_id, current_target_set)
                        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
                        dest_lbl_path = dest_lbl_dir / f"{img_fname_stem_in_proj}.txt"
                        
                        try:
                            parsed_label_lines = []
                            with open(found_label_path_in_zip, 'r') as f_lbl_zip:
                                for line in f_lbl_zip:
                                    parts = line.strip().split()
                                    if len(parts) == 5:
                                        class_id_str_or_name = parts[0]
                                        try:
                                            class_id = int(class_id_str_or_name)
                                            temp_id_to_name = {v: k for k, v in metadata.class_mapping.items()}
                                            if class_id not in temp_id_to_name:
                                                logger.warning(f"ID de classe '{class_id}' dans {found_label_path_in_zip.name} non trouvé dans le mapping. Ignoré.")
                                                continue
                                        except ValueError:
                                            class_name = class_id_str_or_name
                                            if class_name not in metadata.class_mapping:
                                                new_cls_id = len(metadata.class_mapping)
                                                metadata.class_mapping[class_name] = new_cls_id
                                                if class_name not in metadata.classes: metadata.classes.append(class_name)
                                                new_classes_this_upload.add(class_name)
                                            class_id = metadata.class_mapping[class_name]

                                        x_c, y_c, w, h = map(float, parts[1:])
                                        parsed_label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                            if parsed_label_lines:
                                with open(dest_lbl_path, "w") as f_dest_lbl:
                                    f_dest_lbl.write("\n".join(parsed_label_lines))
                                labels_found_count += 1
                        except Exception as e_label_proc:
                            logger.error(f"Erreur traitement label {found_label_path_in_zip}: {e_label_proc}", exc_info=True)
                            if dest_lbl_path.exists(): dest_lbl_path.unlink(missing_ok=True)

        metadata.classes = sorted(list(set(metadata.classes)))
        updated_metadata = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, updated_metadata)
        update_project_dataset_yaml(project_id)

        msg = f"{images_added_count} images et {labels_found_count} labels traités. "
        if new_classes_this_upload:
            msg += f"Nouvelles classes ajoutées: {', '.join(sorted(list(new_classes_this_upload)))}. "
        
        return UploadDatasetZipInfo(
            message=msg, project_id=project_id, images_added=images_added_count,
            labels_found=labels_found_count, new_classes_added=sorted(list(new_classes_this_upload)),
            updated_project_info=updated_metadata
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors du traitement du ZIP: {str(e)}")
    finally:
        if temp_zip_path.exists(): temp_zip_path.unlink(missing_ok=True)
        if temp_extract_path.exists(): shutil.rmtree(temp_extract_path, ignore_errors=True)

@app.post("/finetune_projects", response_model=ProjectInfo, status_code=201, tags=["Projects"])
async def create_finetune_project(project_data: ProjectCreateRequest):
    project_id = str(uuid.uuid4())
    project_dir = get_project_dir(project_id)
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "images").mkdir(exist_ok=True); (project_dir / "labels").mkdir(exist_ok=True)
        for s in ["train", "val", "test"]:
            (project_dir / "images" / s).mkdir(exist_ok=True); (project_dir / "labels" / s).mkdir(exist_ok=True)
        metadata = ProjectInfo(
            project_id=project_id, name=project_data.name, description=project_data.description,
            created_at=datetime.now(timezone.utc), stats=ProjectDatasetStats()
        )
        save_project_metadata(project_id, metadata)
        return metadata
    except Exception as e:
        if project_dir.exists(): shutil.rmtree(project_dir)
        raise HTTPException(status_code=500, detail=f"Erreur serveur création projet: {str(e)}")

def make_aware_datetime(dt: datetime) -> datetime:
    """Ensure a datetime object is timezone-aware (assumes UTC if naive)."""
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

@app.get("/finetune_projects", response_model=List[ProjectInfo], tags=["Projects"])
async def list_finetune_projects():
    projects = []
    if not FINETUNE_PROJECTS_DIR.exists(): return []
    for p_dir_item in FINETUNE_PROJECTS_DIR.iterdir():
        if p_dir_item.is_dir():
            metadata = load_project_metadata(p_dir_item.name)
            if metadata:
                metadata_updated = _update_project_image_counts_and_stats(p_dir_item.name, metadata)
                if metadata != metadata_updated:
                    save_project_metadata(p_dir_item.name, metadata_updated)
                projects.append(metadata_updated)
    return sorted(projects, key=lambda p: make_aware_datetime(p.created_at), reverse=True)

@app.get("/finetune_projects/{project_id}", response_model=ProjectInfo, tags=["Projects"])
async def get_finetune_project_details(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    metadata_updated = _update_project_image_counts_and_stats(project_id, metadata)
    if metadata_updated != metadata:
        save_project_metadata(project_id, metadata_updated)
    return metadata_updated

@app.put("/finetune_projects/{project_id}", response_model=ProjectInfo, tags=["Projects"])
async def update_finetune_project_details(project_id: str, project_update_data: ProjectCreateRequest):
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    metadata.name = project_update_data.name
    metadata.description = project_update_data.description
    save_project_metadata(project_id, metadata)
    return metadata

@app.delete("/finetune_projects/{project_id}", status_code=204, tags=["Projects"])
async def delete_finetune_project(project_id: str):
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    try:
        shutil.rmtree(project_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur suppression projet: {str(e)}")
    return Response(status_code=204)

@app.post("/finetune_projects/{project_id}/images", response_model=ProjectInfo, tags=["Projects Data"])
async def add_image_to_finetune_project(project_id: str, data: AddImageToProjectRequest):
    metadata = load_project_metadata(project_id)
    if not metadata: raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    temp_image_path = TEMP_UPLOAD_DIR / data.temp_image_id
    if not temp_image_path.exists(): raise HTTPException(status_code=404, detail=f"Image temp '{data.temp_image_id}' non trouvée.")

    target_set_dir_img = get_project_images_dir(project_id, data.target_set)
    target_set_dir_lbl = get_project_labels_dir(project_id, data.target_set)
    target_set_dir_img.mkdir(parents=True, exist_ok=True)
    target_set_dir_lbl.mkdir(parents=True, exist_ok=True)

    img_fname_stem_in_proj = f"{Path(data.original_filename).stem}_{uuid.uuid4().hex[:8]}"
    img_fname_in_proj = f"{img_fname_stem_in_proj}{temp_image_path.suffix}"
    dest_img_path = target_set_dir_img / img_fname_in_proj
    dest_lbl_path = target_set_dir_lbl / f"{img_fname_stem_in_proj}.txt"

    try:
        shutil.copy(temp_image_path, dest_img_path)
        class_mapping = metadata.class_mapping
        for ann in data.annotations:
            if ann.label not in class_mapping:
                new_id = len(class_mapping)
                class_mapping[ann.label] = new_id
        metadata.classes = sorted(class_mapping.keys())
        metadata.class_mapping = class_mapping

        img_h, img_w = cv2.imread(str(dest_img_path)).shape[:2]

        label_lines = []
        for ann in data.annotations:
            class_id = class_mapping[ann.label]
            norm_x1, norm_x2 = min(ann.x1, ann.x2) / img_w, max(ann.x1, ann.x2) / img_w
            norm_y1, norm_y2 = min(ann.y1, ann.y2) / img_h, max(ann.y1, ann.y2) / img_h
            x_c, y_c, w, h = (norm_x1 + norm_x2) / 2, (norm_y1 + norm_y2) / 2, norm_x2 - norm_x1, norm_y2 - norm_y1
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        
        with open(dest_lbl_path, "w") as f: f.write("\n".join(label_lines))

        metadata_updated = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, metadata_updated)
        update_project_dataset_yaml(project_id)
        return metadata_updated
    except Exception as e:
        if dest_img_path.exists(): dest_img_path.unlink(missing_ok=True)
        if dest_lbl_path.exists(): dest_lbl_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur ajout image: {str(e)}")

@app.post("/finetune_projects/{project_id}/split_dataset", response_model=ProjectInfo, tags=["Projects Data"])
async def split_project_dataset(project_id: str, split_request: SplitDatasetRequest):
    metadata = load_project_metadata(project_id)
    if not metadata: raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    
    if not np.isclose(split_request.train_ratio + split_request.val_ratio + split_request.test_ratio, 1.0):
        raise HTTPException(status_code=400, detail="La somme des ratios doit être égale à 1.0")

    if metadata.is_split:
        logger.info(f"Projet '{project_id}' déjà splitté. Re-consolidation avant division...")
        for s_name in ["train", "val", "test"]:
            s_img_dir = get_project_images_dir(project_id, s_name)
            s_lbl_dir = get_project_labels_dir(project_id, s_name)
            target_root_img_dir = get_project_images_dir(project_id)
            target_root_lbl_dir = get_project_labels_dir(project_id)
            if s_img_dir.exists():
                for item in s_img_dir.iterdir(): shutil.move(str(item), str(target_root_img_dir / item.name))
            if s_lbl_dir.exists():
                for item in s_lbl_dir.iterdir(): shutil.move(str(item), str(target_root_lbl_dir / item.name))
        metadata.is_split = False

    source_images_dir = get_project_images_dir(project_id)
    source_labels_dir = get_project_labels_dir(project_id)
    all_image_files = [f for f in source_images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']]
    if not all_image_files: raise HTTPException(status_code=400, detail="Aucune image à diviser.")

    if split_request.shuffle: random.shuffle(all_image_files)
    num_total = len(all_image_files)
    num_train = int(split_request.train_ratio * num_total)
    num_val = min(int(split_request.val_ratio * num_total), num_total - num_train)
    splits = {"train": all_image_files[:num_train], "val": all_image_files[num_train:num_train+num_val], "test": all_image_files[num_train+num_val:]}

    try:
        for set_name, image_files in splits.items():
            dest_img_dir = get_project_images_dir(project_id, set_name); dest_img_dir.mkdir(exist_ok=True)
            dest_lbl_dir = get_project_labels_dir(project_id, set_name); dest_lbl_dir.mkdir(exist_ok=True)
            for img_file in image_files:
                shutil.move(str(img_file), str(dest_img_dir / img_file.name))
                label_file = source_labels_dir / f"{img_file.stem}.txt"
                if label_file.exists(): shutil.move(str(label_file), str(dest_lbl_dir / label_file.name))
        
        metadata_updated = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, metadata_updated)
        update_project_dataset_yaml(project_id)
        return metadata_updated
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur division dataset: {str(e)}.")

@app.post("/finetune_projects/{project_id}/export_coco", tags=["Projects Data"])
async def export_project_to_coco(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata or metadata.image_count == 0:
        raise HTTPException(status_code=400, detail=f"Projet '{project_id}' non trouvé ou sans images.")

    coco_output = {
        "info": {"description": f"COCO export: {metadata.name}"},
        "licenses": [], "images": [], "annotations": [],
        "categories": [{"id": cid, "name": name} for name, cid in metadata.class_mapping.items()]
    }
    
    annotation_id, image_id_coco = 1, 1
    sets_to_export = [None] if not metadata.is_split else [s for s in ["train", "val"] if getattr(metadata, f"{s}_image_count") > 0]

    for subset in sets_to_export:
        image_dir = get_project_images_dir(project_id, subset)
        label_dir = get_project_labels_dir(project_id, subset)
        if not image_dir.exists(): continue

        for img_file in image_dir.glob("*.*"):
            if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']: continue
            try:
                pil_img = Image.open(img_file)
                width, height = pil_img.size
            except Exception: continue

            coco_output["images"].append({"id": image_id_coco, "width": width, "height": height, "file_name": f"{subset or 'root'}/{img_file.name}"})
            label_file = label_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f_label:
                    for line in f_label:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cid, xcn, ycn, wn, hn = int(parts[0]), *map(float, parts[1:])
                            aw, ah = wn * width, hn * height
                            axm, aym = (xcn * width) - (aw / 2), (ycn * height) - (ah / 2)
                            coco_output["annotations"].append({
                                "id": annotation_id, "image_id": image_id_coco, "category_id": cid,
                                "bbox": [round(axm,2), round(aym,2), round(aw,2), round(ah,2)],
                                "area": round(aw * ah, 2), "iscrowd": 0
                            })
                            annotation_id += 1
            image_id_coco += 1
            
    filename = f"{metadata.name.replace(' ', '_')}_coco.json"
    return JSONResponse(content=coco_output, headers={"Content-Disposition": f"attachment; filename={filename}"})

def _run_finetune_background_task(task: TaskStatus, project_id: str, base_model_path_str: str, ft_config: FineTuneRunConfig, run_name_override: Optional[str], save_period: int):
    _update_task_status(task.task_id, "running", "Préparation fine-tuning...", progress=0.01)
    try:
        project_metadata = load_project_metadata(project_id)
        if not project_metadata: raise ValueError(f"Métadonnées projet {project_id} non trouvées.")
        update_project_dataset_yaml(project_id)
        project_metadata = load_project_metadata(project_id)
        dataset_yaml_path_str = project_metadata.dataset_yaml_path
        if not dataset_yaml_path_str or not Path(dataset_yaml_path_str).exists():
            raise ValueError(f"dataset.yaml pour projet {project_id} non trouvé.")
        
        base_model_p = Path(base_model_path_str)
        if not base_model_p.is_file(): raise FileNotFoundError(f"Modèle base '{base_model_p}' non trouvé.")

        model_to_train = YOLO(str(base_model_p))

        def progress_callback(trainer):
            progress_value = ((trainer.epoch + 1) / trainer.epochs * 0.9 + 0.05) if trainer.epochs > 0 else 0.05
            _update_task_status(
                task.task_id, "running", 
                f"Epoch {trainer.epoch + 1}/{trainer.epochs} terminée.", 
                progress=progress_value
            )
        model_to_train.add_callback("on_epoch_end", progress_callback)
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        project_name_slug = "".join(c if c.isalnum() else '_' for c in project_metadata.name.lower())[:20]
        effective_run_name = run_name_override or f"ft_{project_name_slug}_{base_model_p.stem}_{timestamp}"
        effective_run_name = "".join(c for c in effective_run_name if c.isalnum() or c in ['_', '-'])
        if get_run_dir(effective_run_name).exists():
            effective_run_name = f"{effective_run_name}_{uuid.uuid4().hex[:4]}"

        _update_task_status(task.task_id, "running", f"Entraînement sur {ft_config.device}. Run: {effective_run_name}", progress=0.05)
        
        train_args = ft_config.model_dump(exclude={"augmentations"})
        train_args.update(ft_config.augmentations.model_dump())
        train_args.update({
            "data": str(dataset_yaml_path_str), "project": str(FINETUNED_MODELS_OUTPUT_DIR),
            "name": effective_run_name, "exist_ok": True, "save_period": save_period
        })
        
        logger.info(f"Tâche FT [{task.task_id}]: Démarrage train avec args: {train_args}")
        results = model_to_train.train(**train_args)
        _update_task_status(task.task_id, "running", "Finalisation du run...", progress=0.95)
        
        actual_run_output_dir = Path(results.save_dir)
        final_run_name = actual_run_output_dir.name
        
        best_model_path = actual_run_output_dir / "weights" / "best.pt"
        last_model_path = actual_run_output_dir / "weights" / "last.pt"
        model_name_pt, final_model_path = ("best.pt", best_model_path) if best_model_path.exists() else ("last.pt", last_model_path)
        if not final_model_path or not final_model_path.exists(): raise FileNotFoundError("best.pt ou last.pt non trouvés.")
        
        artifacts = [RunArtifact(name=f.name, path_on_server_relative_to_run_dir=str(f.relative_to(actual_run_output_dir)), type=f.suffix.lstrip('.')) for f in actual_run_output_dir.glob("**/*") if f.is_file()]
        
        run_metadata = FineTunedModelInfo(
            run_name=final_run_name, model_name_pt=model_name_pt, model_path_abs=str(final_model_path.resolve()),
            model_path_relative_to_home=str(final_model_path.relative_to(HOME_DIR)), project_id_source=project_id,
            base_model_used=base_model_p.name, training_date=datetime.now(timezone.utc), run_config_used=ft_config.model_dump(),
            final_metrics=results.results_dict, artifacts=artifacts, output_dir_abs=str(actual_run_output_dir.resolve()),
            output_dir_relative_to_home=str(actual_run_output_dir.relative_to(HOME_DIR)), best_epoch=getattr(results, 'epoch', None)
        )
        with open(get_run_metadata_path(final_run_name), "w") as f_meta: json.dump(run_metadata.model_dump(mode='json'), f_meta, indent=4)
        _update_task_status(task.task_id, "completed", f"Terminé. Modèle: {model_name_pt}", result=run_metadata.model_dump(mode='json'), progress=1.0)
    except Exception as e:
        logger.error(f"Erreur tâche FT [{task.task_id}]: {e}", exc_info=True)
        _update_task_status(task.task_id, "error", f"Erreur: {str(e)}", progress=task.progress or 0.0)

@app.post("/finetune", response_model=TaskStatusResponse, tags=["Fine-Tuning"])
async def start_finetune_model_on_project(request_data: FineTuneRequest, background_tasks: BackgroundTasks):
    if not get_project_dir(request_data.project_id).exists(): raise HTTPException(status_code=404, detail=f"Projet '{request_data.project_id}' non trouvé.")
    base_model_path = Path(request_data.base_model_path)
    if not base_model_path.is_absolute():
        base_model_path = (HOME_DIR / base_model_path).resolve()
    if not base_model_path.is_file():
        raise HTTPException(status_code=404, detail=f"Modèle base '{request_data.base_model_path}' non trouvé.")
    
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type="finetune")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(_run_finetune_background_task, new_task, request_data.project_id, str(base_model_path), request_data.run_config, request_data.run_name, request_data.save_period)
    return TaskStatusResponse(**new_task.model_dump())

@app.get("/finetune_tasks", response_model=List[TaskStatusResponse], tags=["Fine-Tuning Tasks"])
async def list_all_tasks():
    tasks_db = load_tasks_db()
    return sorted([TaskStatusResponse(**task.model_dump()) for task in tasks_db.values()], key=lambda t: make_aware_datetime(t.created_at), reverse=True)

@app.get("/finetune_tasks/{task_id}", response_model=TaskStatusResponse, tags=["Fine-Tuning Tasks"])
async def get_task_status(task_id: str):
    task = load_tasks_db().get(task_id)
    if not task: raise HTTPException(status_code=404, detail=f"Tâche ID '{task_id}' non trouvée.")
    return TaskStatusResponse(**task.model_dump())

@app.get("/finetuned_models", response_model=List[FineTunedModelInfo], tags=["Fine-Tuned Models"])
async def list_finetuned_models():
    tuned_models_info = []
    if not FINETUNED_MODELS_OUTPUT_DIR.exists(): return []
    for run_dir in FINETUNED_MODELS_OUTPUT_DIR.iterdir():
        if run_dir.is_dir():
            meta_path = get_run_metadata_path(run_dir.name)
            if meta_path.exists():
                try:
                    with open(meta_path, 'r') as f_meta: data = json.load(f_meta)
                    if isinstance(data.get("training_date"), str):
                        data["training_date"] = datetime.fromisoformat(data["training_date"].replace('Z', '+00:00'))
                    info = FineTunedModelInfo(**data)
                    if Path(info.model_path_abs).exists():
                        tuned_models_info.append(info)
                except Exception: continue
    return sorted(tuned_models_info, key=lambda m: make_aware_datetime(m.training_date), reverse=True)

@app.get("/finetuned_models/{run_name}", response_model=FineTunedModelInfo, tags=["Fine-Tuned Models"])
async def get_finetuned_model_details(run_name: str):
    meta_path = get_run_metadata_path(run_name)
    if not meta_path.exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    try:
        with open(meta_path, 'r') as f: data = json.load(f)
        if isinstance(data.get("training_date"), str):
            data["training_date"] = datetime.fromisoformat(data["training_date"].replace('Z', '+00:00'))
        return FineTunedModelInfo(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lecture métadonnées: {str(e)}")

@app.put("/finetuned_models/{run_name}/notes", response_model=FineTunedModelInfo, tags=["Fine-Tuned Models"])
async def update_finetuned_model_notes(run_name: str, notes: str = Body(..., embed=True, max_length=1000)):
    meta_path = get_run_metadata_path(run_name)
    if not meta_path.exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    try:
        with open(meta_path, 'r') as f: data = json.load(f)
        if isinstance(data.get("training_date"), str):
            data["training_date"] = datetime.fromisoformat(data["training_date"].replace('Z', '+00:00'))
        model_info = FineTunedModelInfo(**data)
        model_info.notes = notes
        with open(meta_path, 'w') as f: json.dump(model_info.model_dump(mode='json'), f, indent=4)
        return model_info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur màj notes: {str(e)}")

@app.get("/finetuned_models/{run_name}/artifacts/{artifact_filename:path}", tags=["Fine-Tuned Models"])
async def get_run_artifact(run_name: str, artifact_filename: str):
    run_dir = get_run_dir(run_name)
    if not run_dir.is_dir(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    artifact_path = (run_dir / artifact_filename).resolve()
    if not str(artifact_path).startswith(str(run_dir.resolve())) or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artefact '{artifact_filename}' non trouvé.")
    return FileResponse(path=artifact_path, filename=artifact_path.name)

@app.get("/download_model/{model_path_b64}", tags=["Models"])
async def download_any_model_by_b64_path(model_path_b64: str):
    try:
        relative_path_str = base64.urlsafe_b64decode(model_path_b64.encode() + b'==').decode()
    except Exception:
        raise HTTPException(status_code=400, detail="Chemin encodé invalide.")
    
    full_path = (HOME_DIR / Path(relative_path_str)).resolve()
    if not full_path.is_file() or not str(full_path).startswith(str(HOME_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Fichier non trouvé ou accès non autorisé.")
    return FileResponse(path=full_path, filename=full_path.name, media_type='application/octet-stream')

def _run_export_model_task(task: TaskStatus, run_name: str, export_config: ExportFormat):
    _update_task_status(task.task_id, "running", f"Préparation export {export_config.format}...", progress=0.01)
    try:
        run_meta_path = get_run_metadata_path(run_name)
        if not run_meta_path.exists(): raise FileNotFoundError("Métadonnées run non trouvées.")
        with open(run_meta_path, 'r') as f: data = json.load(f)
        if isinstance(data.get("training_date"), str):
            data["training_date"] = datetime.fromisoformat(data["training_date"].replace('Z', '+00:00'))
        run_info = FineTunedModelInfo(**data)
        
        model = YOLO(run_info.model_path_abs)
        export_params = export_config.model_dump(exclude_none=True)
        if export_params.get("imgsz") is None and run_info.run_config_used.get("imgsz"):
            export_params["imgsz"] = run_info.run_config_used["imgsz"]

        _update_task_status(task.task_id, "running", f"Exportation...", progress=0.3)
        exported_path = Path(model.export(**export_params))
        if not exported_path.exists(): raise FileNotFoundError("Fichier exporté non créé.")
        
        new_artifact = RunArtifact(
            name=exported_path.name,
            path_on_server_relative_to_run_dir=str(exported_path.relative_to(get_run_dir(run_name))),
            type=export_config.format
        )
        if not any(a.name == new_artifact.name for a in run_info.artifacts):
            run_info.artifacts.append(new_artifact)
            with open(run_meta_path, "w") as f_meta: json.dump(run_info.model_dump(mode='json'), f_meta, indent=4)

        result = {"exported_model_name": exported_path.name, "run_name_source": run_name}
        _update_task_status(task.task_id, "completed", f"Export terminé: {exported_path.name}", result=result, progress=1.0)
    except Exception as e:
        _update_task_status(task.task_id, "error", f"Erreur export: {str(e)}")

@app.post("/finetuned_models/{run_name}/export", response_model=TaskStatusResponse, tags=["Fine-Tuned Models Export"])
async def export_finetuned_model(run_name: str, export_config: ExportFormat, background_tasks: BackgroundTasks):
    if not get_run_dir(run_name).exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type=f"export_{export_config.format}")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(_run_export_model_task, new_task, run_name, export_config)
    return TaskStatusResponse(**new_task.model_dump())

def _run_hpo_task(task: TaskStatus, hpo_request: HyperparameterTuneRequest):
    _update_task_status(task.task_id, "running", "Préparation HPO...", progress=0.01)
    try:
        project_metadata = load_project_metadata(hpo_request.project_id)
        if not project_metadata: raise ValueError("Métadonnées projet non trouvées.")
        update_project_dataset_yaml(hpo_request.project_id)
        dataset_yaml_path = load_project_metadata(hpo_request.project_id).dataset_yaml_path
        if not dataset_yaml_path: raise ValueError("dataset.yaml non trouvé.")
        
        base_model_path = (HOME_DIR / hpo_request.base_model_path).resolve()
        if not base_model_path.is_file(): raise FileNotFoundError("Modèle base non trouvé.")
        
        model = YOLO(str(base_model_path))
        hpo_run_name = f"hpo_{project_metadata.name.replace(' ','_')}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        tune_args = hpo_request.base_run_config.model_dump(exclude={"augmentations"})
        tune_args.update(hpo_request.base_run_config.augmentations.model_dump())
        tune_args.update({"data": dataset_yaml_path, "iterations": hpo_request.num_trials, "project": str(FINETUNED_MODELS_OUTPUT_DIR), "name": hpo_run_name})

        _update_task_status(task.task_id, "running", "HPO en cours...", progress=0.1)
        best_model_path = Path(model.tune(**tune_args))
        if not best_model_path.exists(): raise FileNotFoundError("Meilleur modèle non trouvé après HPO.")
        
        hpo_output_dir = get_run_dir(hpo_run_name)
        best_hyp_path = hpo_output_dir / "hyp.yaml"
        best_config = yaml.safe_load(best_hyp_path.read_text()) if best_hyp_path.exists() else {}
        
        artifacts = [RunArtifact(name=f.name, path_on_server_relative_to_run_dir=str(f.relative_to(hpo_output_dir)), type=f.suffix[1:]) for f in hpo_output_dir.glob("*.*") if f.is_file()]
        run_meta = FineTunedModelInfo(run_name=hpo_run_name, model_name_pt="best.pt", model_path_abs=str(best_model_path.resolve()), model_path_relative_to_home=str(best_model_path.relative_to(HOME_DIR)), project_id_source=hpo_request.project_id, base_model_used=base_model_path.name, training_date=datetime.now(timezone.utc), run_config_used=best_config, artifacts=artifacts, output_dir_abs=str(hpo_output_dir.resolve()), output_dir_relative_to_home=str(hpo_output_dir.relative_to(HOME_DIR)))
        
        with open(get_run_metadata_path(hpo_run_name), "w") as f: json.dump(run_meta.model_dump(mode='json'), f, indent=4)
        _update_task_status(task.task_id, "completed", "HPO terminé.", result=run_meta.model_dump(mode='json'), progress=1.0)
    except Exception as e:
        _update_task_status(task.task_id, "error", f"Erreur HPO: {str(e)}")

@app.post("/hpo_tune", response_model=TaskStatusResponse, tags=["Hyperparameter Optimization"])
async def start_hyperparameter_optimization(hpo_request: HyperparameterTuneRequest, background_tasks: BackgroundTasks):
    if not get_project_dir(hpo_request.project_id).exists(): raise HTTPException(status_code=404, detail=f"Projet '{hpo_request.project_id}' non trouvé.")
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type="hpo_tune")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(_run_hpo_task, new_task, hpo_request)
    return TaskStatusResponse(**new_task.model_dump())

@app.post("/describe_llm", response_model=LLMDescribeResponse, tags=["LLM"])
async def describe_image_with_llm(request: LLMDescribeRequest):
    if not EXTERNAL_LLM_FULL_URL or "your-llm-api" in EXTERNAL_LLM_FULL_URL:
        raise HTTPException(status_code=503, detail="Service LLM non configuré.")
    try:
        response = requests.post(EXTERNAL_LLM_FULL_URL, json=request.model_dump(), headers={"ngrok-skip-browser-warning": "true"}, timeout=120)
        response.raise_for_status()
        data = response.json()
        desc_keys = ["description", "response", "text", "content", "answer"]
        description = next((data[key] for key in desc_keys if key in data and isinstance(data[key], str)), None)
        if description is None and data.get("choices"):
             description = data["choices"][0].get("message", {}).get("content")
        
        if not description: raise ValueError("Format de réponse LLM non reconnu.")
        return LLMDescribeResponse(description=description.strip())
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=504, detail=f"Erreur API LLM: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur serveur (LLM): {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    host_ip = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Démarrage du serveur FastAPI (Advanced ML Suite v1.0.3) sur http://{host_ip}:{port}")
    uvicorn.run(app, host=host_ip, port=port)
