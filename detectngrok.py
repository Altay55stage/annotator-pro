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
import logging # Utiliser le module logging standard

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Body, Query, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, validator, field_validator
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

# --- Modèles Pydantic (étendus) ---
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

# NOUVEAU: Pour l'upload de ZIP
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

    @field_validator('test_ratio', 'train_ratio', 'val_ratio') # Pydantic v2 style
    @classmethod
    def check_ratio_value(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Les ratios doivent être entre 0.0 et 1.0")
        return v
    
    @field_validator('*') # Pour valider la somme après que tous les champs soient lus
    @classmethod
    def check_ratios_sum(cls, values):
        # Note: ce validateur peut ne pas fonctionner comme prévu pour la somme avec Pydantic v2.
        # Il est plus simple de valider dans l'endpoint ou le service.
        # Pour Pydantic v2, utilisez `model_validator` pour les validations inter-champs.
        # train_r = values.data.get('train_ratio') # Accès aux données dans Pydantic v2
        # val_r = values.data.get('val_ratio')
        # test_r = values.data.get('test_ratio')
        # if train_r is not None and val_r is not None and test_r is not None:
        #     if not np.isclose(train_r + val_r + test_r, 1.0):
        #         raise ValueError("La somme des ratios train, val, et test doit être égale à 1.0")
        return values

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
    batch_size: int = DEFAULT_FINETUNE_BATCH_SIZE
    img_size: int = DEFAULT_IMG_SIZE_INT
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
    epochs_range: Optional[Tuple[int, int]] = None
    batch_size_values: Optional[List[int]] = None
    lr0_range: Optional[Tuple[float, float]] = None
    optimizer_values: Optional[List[str]] = None
    num_trials: int = Field(10, ge=1, le=100)

# --- Initialisation FastAPI ---
app = FastAPI(title="YOLO Annotator Pro - Advanced ML Suite", version="1.0.2") # Version up
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- État Global du Serveur ---
current_model_instance: Optional[YOLO] = None
current_model_path_loaded: Optional[Path] = None
current_model_type_loaded: Optional[str] = None

# --- Gestion des tâches (avec correction pour dates) ---
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
                            data[date_key] = datetime.now(timezone.utc) # Fallback
                try:
                    parsed_tasks[task_id] = TaskStatus(**data)
                except Exception as e_pydantic:
                    logger.error(f"Erreur de validation Pydantic pour la tâche {task_id} après parsing date: {e_pydantic}. Données: {data}")
            return parsed_tasks
        except json.JSONDecodeError as e_json:
            logger.error(f"Erreur de décodage JSON pour tasks_db.json: {e_json}. Le fichier est peut-être corrompu. Tentative de renommage en .corrupted.")
            try:
                corrupted_path = TASKS_DB_FILE.with_suffix(f'.corrupted_{datetime.now().strftime("%Y%m%d%H%M%S")}.json')
                TASKS_DB_FILE.rename(corrupted_path)
                logger.info(f"Fichier {TASKS_DB_FILE} renommé en {corrupted_path}")
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
    project_dir = get_project_dir(project_id)
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
                img_path_found = None
                if image_file_stem not in images_processed_for_size:
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                        potential_img_path = img_dir / (image_file_stem + ext)
                        if potential_img_path.exists():
                            img_path_found = potential_img_path
                            break
                    if img_path_found:
                        try:
                            img = Image.open(img_path_found)
                            size_str = f"{img.width}x{img.height}"
                            image_size_distribution[size_str] = image_size_distribution.get(size_str, 0) + 1
                            images_processed_for_size.add(image_file_stem)
                        except Exception: pass

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
        valid_mapping = True
        for name, c_id in sorted_class_items:
            if c_id == expected_id:
                class_names_ordered.append(name)
                expected_id += 1
            else:
                logger.error(f"Rupture ID classes pour {project_id}. Attendu: {expected_id}, Obtenu: {c_id} pour '{name}'.")
                valid_mapping = False; break
        if not valid_mapping: class_names_ordered = []
    
    if not class_names_ordered:
        logger.info(f"Pas de classes valides pour projet {project_id}. YAML indiquera nc=0.")
        yaml_path_current = get_project_dataset_yaml_path(project_id)
        if metadata.image_count == 0 and yaml_path_current.exists():
             yaml_path_current.unlink(missing_ok=True)
             metadata.dataset_yaml_path = None
             save_project_metadata(project_id, metadata)
             return
        if metadata.image_count > 0 and not class_names_ordered:
            logger.warning(f"Projet {project_id} a des images mais pas de classes définies. YAML aura nc=0.")

    yaml_content = {
        "path": str(project_dir_abs),
        "nc": len(class_names_ordered),
        "names": class_names_ordered
    }

    if metadata.image_count > 0 :
        if metadata.is_split:
            if metadata.train_image_count > 0: yaml_content["train"] = str(Path("images") / "train")
            else: logger.warning(f"Projet {project_id} splitté mais 0 image train.")
            if metadata.val_image_count > 0: yaml_content["val"] = str(Path("images") / "val")
            else: logger.info(f"Projet {project_id} splitté mais 0 image val.")
            if metadata.test_image_count > 0: yaml_content["test"] = str(Path("images") / "test")
        else: 
            if metadata.image_count > 0:
                yaml_content["train"] = "images" 
                yaml_content["val"] = "images"
    else:
        logger.info(f"Projet {project_id} n'a aucune image. YAML avec nc={len(class_names_ordered)}.")

    yaml_path = get_project_dataset_yaml_path(project_id)
    try:
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=None)
        logger.info(f"dataset.yaml projet {project_id} màj. Contenu: {yaml_content}")
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
    logger.info(f"Vérification des dossiers de base: HOME_DIR={HOME_DIR}")
    logger.info("Serveur YOLO Annotator Pro (Advanced ML Suite) démarré.")
    yield
    logger.info("Serveur YOLO Annotator Pro (Advanced ML Suite) arrêté.")

app.router.lifespan_context = lifespan

# ... (Endpoints /status, /models, /load_model, /predict sont identiques) ...
@app.get("/status", tags=["General"])
async def get_server_status():
    return {
        "status": "running",
        "timestamp": datetime.now(timezone.utc),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "default_device_for_training": DEFAULT_DEVICE,
        "sahi_available": SAHI_AVAILABLE,
        "home_dir": str(HOME_DIR)
    }

@app.get("/models", response_model=List[ModelInfo], tags=["Models"])
async def list_models_endpoint():
    models_list = []
    if not MODELS_DIR.is_dir():
        logger.error(f"Le dossier des modèles ({MODELS_DIR}) n'a pas été trouvé.")
        raise HTTPException(status_code=500, detail=f"Dossier des modèles ({MODELS_DIR}) non trouvé.")
    try:
        for item_name in os.listdir(MODELS_DIR):
            item_path = MODELS_DIR / item_name
            if item_path.is_file() and item_name.lower().endswith((".pt", ".pth")):
                models_list.append(ModelInfo(name=item_name, path=str(item_path.resolve())))
        if not models_list:
            logger.warning(f"Aucun fichier de modèle (.pt, .pth) trouvé dans {MODELS_DIR}.")
    except Exception as e:
        logger.error(f"Erreur lors du listage des modèles: {e}", exc_info=True)
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
        logger.error(f"Modèle non trouvé: '{model_name_or_path}'.")
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name_or_path}' non trouvé.")

    current_model_instance, current_model_path_loaded, current_model_type_loaded = None, None, None
    try:
        logger.info(f"Tentative de chargement du modèle Ultralytics: {model_to_load_path}")
        loaded_yolo_obj = YOLO(str(model_to_load_path))
        _ = loaded_yolo_obj.predict(np.zeros((32, 32, 3), dtype=np.uint8), verbose=False, device='cpu') 
        current_model_instance = loaded_yolo_obj
        current_model_path_loaded = model_to_load_path
        current_model_type_loaded = "ultralytics"
        logger.info(f"Modèle '{model_to_load_path.name}' chargé avec succès (type: Ultralytics).")
        return {"message": f"Modèle '{model_to_load_path.name}' (Ultralytics) chargé avec succès."}
    except Exception as e_ultralytics:
        logger.warning(f"Échec du chargement du modèle '{model_to_load_path.name}' via Ultralytics: {e_ultralytics}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Échec du chargement du modèle '{model_to_load_path.name}' via Ultralytics: {str(e_ultralytics)}.")

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
        if image_np is None: raise ValueError("Fichier image invalide.")
    except Exception as e:
        logger.error(f"Erreur lecture/sauvegarde image '{original_filename}': {e}", exc_info=True)
        if temp_image_path.exists(): temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Erreur lecture/sauvegarde image: {str(e)}")

    actual_device_to_use = device if device else (current_model_instance.device if hasattr(current_model_instance, 'device') and current_model_instance.device else DEFAULT_DEVICE)
    if actual_device_to_use == 'cuda' and not torch.cuda.is_available(): actual_device_to_use = 'cpu'
    elif actual_device_to_use not in ['cuda', 'cpu'] and not actual_device_to_use.startswith('mps'): actual_device_to_use = 'cpu'
    
    response_detections: List[Detection] = []
    try:
        if current_model_type_loaded == "ultralytics":
            if use_sahi and SAHI_AVAILABLE:
                logger.info(f"Utilisation de SAHI pour prédiction: {current_model_path_loaded.name if current_model_path_loaded else 'Modèle en mémoire'}")
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

        logger.info(f"Prédiction: {len(response_detections)} détections pour '{original_filename}'. Temp ID: {temp_image_id}")
        return DetectionResponse(
            detections=response_detections, image_id=temp_image_id,
            original_filename=original_filename, message="Détection terminée."
        )
    except Exception as e:
        logger.error(f"Erreur détection sur '{original_filename}': {e}", exc_info=True)
        if temp_image_path.exists(): temp_image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur détection: {str(e)}")

# --- Endpoints de Gestion des Projets de Fine-Tuning ---
# ... (La plupart sont identiques à la version précédente)
# AJOUT D'UN ENDPOINT POUR L'UPLOAD DE ZIP DE DATASET
@app.post("/finetune_projects/{project_id}/upload_dataset_zip", response_model=UploadDatasetZipInfo, tags=["Projects Data"])
async def upload_dataset_zip_to_project(
    project_id: str,
    zip_file: UploadFile = File(...),
    default_target_set: str = Form("train"), # 'train', 'val', 'test'
    # auto_split_str: str = Form("false"), # 'true' or 'false'
    # train_ratio_str: Optional[str] = Form("0.7"),
    # val_ratio_str: Optional[str] = Form("0.2"),
    # test_ratio_str: Optional[str] = Form("0.1")
):
    """
    Upload un fichier ZIP contenant des images et/ou des labels pour un projet.
    Le ZIP peut contenir une structure de dossiers:
    - images/train/*.jpg, labels/train/*.txt
    - images/val/*.jpg, labels/val/*.txt
    - images/test/*.jpg, labels/test/*.txt
    Ou des images/labels à la racine du ZIP, qui seront placés dans `default_target_set`.
    """
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")

    if default_target_set not in ["train", "val", "test"]:
        raise HTTPException(status_code=400, detail="`default_target_set` doit être 'train', 'val', ou 'test'.")

    # auto_split = auto_split_str.lower() == "true"
    # split_ratios = None
    # if auto_split:
    #     try:
    #         split_ratios = SplitDatasetRequest(
    #             train_ratio=float(train_ratio_str),
    #             val_ratio=float(val_ratio_str),
    #             test_ratio=float(test_ratio_str)
    #         )
    #     except ValueError as e:
    #         raise HTTPException(status_code=400, detail=f"Ratios de split invalides: {e}")

    temp_zip_path = TEMP_UPLOAD_DIR / f"{project_id}_{uuid.uuid4().hex}.zip"
    temp_extract_path = TEMP_UPLOAD_DIR / f"extracted_{project_id}_{uuid.uuid4().hex}"

    try:
        # Sauvegarder le ZIP
        with open(temp_zip_path, "wb") as f_zip:
            shutil.copyfileobj(zip_file.file, f_zip)
        
        # Extraire le ZIP
        temp_extract_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_path)
        
        logger.info(f"ZIP uploadé et extrait dans {temp_extract_path} pour projet {project_id}")

        images_added_count = 0
        labels_found_count = 0
        new_classes_this_upload = set()

        # Parcourir les fichiers extraits
        for root, _, files in os.walk(temp_extract_path):
            relative_root = Path(root).relative_to(temp_extract_path) # Chemin relatif dans le ZIP

            # Déterminer le target_set basé sur la structure du ZIP ou le défaut
            current_target_set = default_target_set
            path_parts = list(relative_root.parts) # ex: ['images', 'train'] ou ['train']

            if len(path_parts) > 0:
                # Si le premier dossier est 'images' ou 'labels', regarder le suivant pour train/val/test
                if path_parts[0].lower() in ["images", "labels"] and len(path_parts) > 1 and path_parts[1].lower() in ["train", "val", "test"]:
                    current_target_set = path_parts[1].lower()
                # Si le premier dossier est directement train/val/test
                elif path_parts[0].lower() in ["train", "val", "test"]:
                     current_target_set = path_parts[0].lower()


            for filename in files:
                file_path_in_zip = Path(root) / filename
                
                # Traiter les images
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                    original_img_name = Path(filename) # Garder le nom original
                    img_fname_stem_in_proj = f"{original_img_name.stem}_{uuid.uuid4().hex[:8]}"
                    img_fname_in_proj = f"{img_fname_stem_in_proj}{original_img_name.suffix}"
                    
                    dest_img_dir = get_project_images_dir(project_id, current_target_set)
                    dest_img_dir.mkdir(parents=True, exist_ok=True)
                    dest_img_path = dest_img_dir / img_fname_in_proj
                    
                    shutil.copy(file_path_in_zip, dest_img_path)
                    images_added_count += 1
                    
                    # Chercher le fichier label correspondant (avec le nom *original* de l'image)
                    original_label_filename = f"{original_img_name.stem}.txt"
                    
                    # Le label peut être à côté de l'image dans le zip, ou dans un dossier labels/ parallèle
                    label_path_in_zip_alongside = file_path_in_zip.with_name(original_label_filename)
                    
                    # Construire le chemin vers un dossier labels/ parallèle dans la structure du ZIP
                    # ex: si image est dans `extracted_zip/images/train/img.jpg`
                    # label pourrait être dans `extracted_zip/labels/train/img.txt`
                    label_path_in_zip_parallel = None
                    if "images" in path_parts:
                        parallel_label_path_parts = [p if p != "images" else "labels" for p in path_parts]
                        label_path_in_zip_parallel = temp_extract_path / Path(*parallel_label_path_parts) / original_label_filename
                    elif len(path_parts) == 1 and path_parts[0] in ["train", "val", "test"]: # si image est dans extracted_zip/train/img.jpg
                        label_path_in_zip_parallel = temp_extract_path / "labels" / path_parts[0] / original_label_filename


                    found_label_path_in_zip = None
                    if label_path_in_zip_alongside.exists():
                        found_label_path_in_zip = label_path_in_zip_alongside
                    elif label_path_in_zip_parallel and label_path_in_zip_parallel.exists():
                        found_label_path_in_zip = label_path_in_zip_parallel
                    
                    if found_label_path_in_zip:
                        dest_lbl_dir = get_project_labels_dir(project_id, current_target_set)
                        dest_lbl_dir.mkdir(parents=True, exist_ok=True)
                        dest_lbl_path = dest_lbl_dir / f"{img_fname_stem_in_proj}.txt" # Label utilise le nouveau stem unique
                        
                        # Lire le label, parser les class_ids, mettre à jour class_mapping
                        try:
                            img_for_size = cv2.imread(str(dest_img_path))
                            if img_for_size is None: raise ValueError("Image copiée non lisible pour dimensions.")
                            img_h, img_w = img_for_size.shape[:2]
                            
                            parsed_label_lines = []
                            with open(found_label_path_in_zip, 'r') as f_lbl_zip:
                                for line_num, line in enumerate(f_lbl_zip):
                                    parts = line.strip().split()
                                    if len(parts) == 5: # Format YOLO: class_id x y w h
                                        class_id_str_or_name = parts[0] # Peut être un ID ou un nom de classe
                                        
                                        # Essayer de convertir en int si c'est un ID, sinon c'est un nom de classe
                                        try:
                                            class_id = int(class_id_str_or_name)
                                            # Si c'est un ID, il faut le mapper à un nom si possible, ou l'ignorer si on attend des noms
                                            # Pour l'instant, on suppose que si c'est un ID, il est déjà conforme au mapping du projet
                                            # (ce qui est peu probable pour un import externe)
                                            # -> Pour un import robuste, il faudrait un mapping fourni par l'utilisateur ou un format qui inclut les noms.
                                            # -> Simplification: Si c'est un int, on essaie de trouver son nom dans le mapping existant.
                                            # -> Sinon, on traite comme un nom de classe.
                                            
                                            temp_id_to_name = {v: k for k, v in metadata.class_mapping.items()}
                                            class_name = temp_id_to_name.get(class_id)
                                            if not class_name:
                                                logger.warning(f"ID de classe '{class_id}' dans {found_label_path_in_zip.name} (ligne {line_num+1}) non trouvé dans le mapping du projet. Ignoré.")
                                                continue # Ignorer cette annotation si l'ID n'est pas mappé

                                        except ValueError: # Ce n'est pas un int, c'est un nom de classe
                                            class_name = class_id_str_or_name
                                            if class_name not in metadata.class_mapping:
                                                new_cls_id = len(metadata.class_mapping)
                                                metadata.class_mapping[class_name] = new_cls_id
                                                if class_name not in metadata.classes: metadata.classes.append(class_name)
                                                new_classes_this_upload.add(class_name)
                                            class_id = metadata.class_mapping[class_name]

                                        # Les coordonnées x, y, w, h sont supposées être déjà normalisées si c'est un label YOLO
                                        # Si elles étaient absolues, il faudrait les normaliser ici.
                                        # On assume qu'elles sont déjà normalisées.
                                        x_c, y_c, w, h = map(float, parts[1:])
                                        if not (0<=x_c<=1 and 0<=y_c<=1 and 0<=w<=1 and 0<=h<=1):
                                            logger.warning(f"Annotation hors limites dans {found_label_path_in_zip.name} (ligne {line_num+1}). Elle sera écrite.")
                                        parsed_label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
                                    else:
                                        logger.warning(f"Ligne de label malformée dans {found_label_path_in_zip.name} (ligne {line_num+1}): {line.strip()}")

                            if parsed_label_lines:
                                with open(dest_lbl_path, "w") as f_dest_lbl:
                                    f_dest_lbl.write("\n".join(parsed_label_lines))
                                labels_found_count += 1
                        except Exception as e_label_proc:
                            logger.error(f"Erreur traitement label {found_label_path_in_zip} pour image {dest_img_path}: {e_label_proc}", exc_info=True)
                            if dest_lbl_path.exists(): dest_lbl_path.unlink(missing_ok=True) # Nettoyer label partiel

        metadata.classes = sorted(list(set(metadata.classes))) # Assurer unicité et ordre

        # Mettre à jour les métadonnées finales du projet
        updated_metadata = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, updated_metadata)
        update_project_dataset_yaml(project_id)

        msg = f"{images_added_count} images et {labels_found_count} labels traités depuis le ZIP. "
        if new_classes_this_upload:
            msg += f"Nouvelles classes ajoutées: {', '.join(sorted(list(new_classes_this_upload)))}. "
        
        logger.info(msg + f"Projet {project_id} mis à jour.")
        return UploadDatasetZipInfo(
            message=msg,
            project_id=project_id,
            images_added=images_added_count,
            labels_found=labels_found_count,
            new_classes_added=sorted(list(new_classes_this_upload)),
            updated_project_info=updated_metadata
        )

    except Exception as e:
        logger.error(f"Erreur lors de l'upload du dataset ZIP pour projet {project_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors du traitement du ZIP: {str(e)}")
    finally:
        # Nettoyage des fichiers temporaires
        if temp_zip_path.exists(): temp_zip_path.unlink(missing_ok=True)
        if temp_extract_path.exists(): shutil.rmtree(temp_extract_path, ignore_errors=True)


@app.post("/finetune_projects", response_model=ProjectInfo, status_code=201, tags=["Projects"])
async def create_finetune_project(project_data: ProjectCreateRequest):
    project_id = str(uuid.uuid4())
    project_dir = get_project_dir(project_id)
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        for s in ["train", "val", "test"]: # Créer aussi les dossiers images/labels de base
            (project_dir / "images" / s).mkdir(parents=True, exist_ok=True)
            (project_dir / "labels" / s).mkdir(parents=True, exist_ok=True)
            (project_dir / "images").mkdir(parents=True, exist_ok=True) # Dossier racine images
            (project_dir / "labels").mkdir(parents=True, exist_ok=True) # Dossier racine labels
        metadata = ProjectInfo(
            project_id=project_id, name=project_data.name, description=project_data.description,
            created_at=datetime.now(timezone.utc), stats=ProjectDatasetStats() # Initialiser avec stats vides
        )
        save_project_metadata(project_id, metadata)
        logger.info(f"Projet '{project_data.name}' (ID: {project_id}) créé.")
        return metadata
    except Exception as e:
        logger.error(f"Erreur création projet '{project_data.name}': {e}", exc_info=True)
        if project_dir.exists(): shutil.rmtree(project_dir)
        raise HTTPException(status_code=500, detail=f"Erreur serveur création projet: {str(e)}")

# ... (le reste des endpoints Projects, Fine-Tuning, Tasks, Models, Export, HPO, LLM reste identique à la version v1.0.1)
# S'assurer que les appels à load_project_metadata, save_project_metadata,
# _update_project_image_counts_and_stats, et update_project_dataset_yaml
# utilisent les versions corrigées/mises à jour de ce fichier.
# La logique de `_run_finetune_background_task` et d'autres tâches de fond
# devrait bénéficier des améliorations dans la gestion des métadonnées et du YAML.

@app.get("/finetune_projects", response_model=List[ProjectInfo], tags=["Projects"])
async def list_finetune_projects():
    projects = []
    if not FINETUNE_PROJECTS_DIR.exists(): return []
    for p_dir_name in FINETUNE_PROJECTS_DIR.iterdir():
        if p_dir_name.is_dir():
            try: uuid.UUID(p_dir_name.name) # Valider que c'est un UUID
            except ValueError: continue
            metadata = load_project_metadata(p_dir_name.name)
            if metadata:
                # Mettre à jour les comptes et stats avant de renvoyer (peut être coûteux)
                # Pourrait être fait moins fréquemment ou à la demande explicite
                metadata_updated = _update_project_image_counts_and_stats(p_dir_name.name, metadata)
                # Sauvegarder si des changements ont été faits par _update_project_image_counts_and_stats
                if metadata != metadata_updated : save_project_metadata(p_dir_name.name, metadata_updated)
                projects.append(metadata_updated)
    return sorted(projects, key=lambda p: p.created_at, reverse=True)


@app.get("/finetune_projects/{project_id}", response_model=ProjectInfo, tags=["Projects"])
async def get_finetune_project_details(project_id: str):
    """Récupère les détails d'un projet de fine-tuning spécifique."""
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet avec ID '{project_id}' non trouvé.")
    # Mettre à jour les comptes et stats avant de renvoyer
    metadata_updated = _update_project_image_counts_and_stats(project_id, metadata)
    if metadata_updated != metadata : # Si des changements ont été faits par _update
        save_project_metadata(project_id, metadata_updated)
    return metadata_updated

@app.put("/finetune_projects/{project_id}", response_model=ProjectInfo, tags=["Projects"])
async def update_finetune_project_details(project_id: str, project_update_data: ProjectCreateRequest):
    """Met à jour le nom ou la description d'un projet."""
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    
    metadata.name = project_update_data.name
    metadata.description = project_update_data.description
    # created_at, project_id, etc. ne changent pas
    
    save_project_metadata(project_id, metadata)
    logger.info(f"Détails du projet '{project_id}' mis à jour.")
    return metadata


@app.delete("/finetune_projects/{project_id}", status_code=204, tags=["Projects"])
async def delete_finetune_project(project_id: str):
    """Supprime un projet de fine-tuning et toutes ses données."""
    project_dir = get_project_dir(project_id)
    if not project_dir.exists():
        raise HTTPException(status_code=404, detail=f"Projet avec ID '{project_id}' non trouvé.")
    try:
        shutil.rmtree(project_dir)
        logger.info(f"Projet '{project_id}' et toutes ses données ont été supprimés.")
    except Exception as e:
        logger.error(f"Erreur lors de la suppression du projet '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur lors de la suppression du projet: {str(e)}")
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
        current_classes_set = set(metadata.classes)
        class_mapping = metadata.class_mapping
        for ann_label in {ann.label for ann in data.annotations}:
            if ann_label not in class_mapping:
                new_id = len(class_mapping)
                class_mapping[ann_label] = new_id
                current_classes_set.add(ann_label)
        metadata.classes = sorted(list(current_classes_set))
        metadata.class_mapping = class_mapping

        img_read_for_size = cv2.imread(str(dest_img_path))
        if img_read_for_size is None: raise ValueError("Impossible de lire l'image copiée.")
        img_h, img_w = img_read_for_size.shape[:2]

        label_lines = []
        for ann in data.annotations:
            class_id = class_mapping.get(ann.label)
            if class_id is None: logger.error(f"Label '{ann.label}' non trouvé dans mapping."); continue
            norm_x1, norm_x2 = min(ann.x1, ann.x2) / img_w, max(ann.x1, ann.x2) / img_w
            norm_y1, norm_y2 = min(ann.y1, ann.y2) / img_h, max(ann.y1, ann.y2) / img_h
            x_c = (norm_x1 + norm_x2) / 2; y_c = (norm_y1 + norm_y2) / 2
            w = norm_x2 - norm_x1; h = norm_y2 - norm_y1
            if not (0<=x_c<=1 and 0<=y_c<=1 and 0<=w<=1 and 0<=h<=1): logger.warning(f"Annotation hors limites: {ann.label} sur {img_fname_in_proj}")
            label_lines.append(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")
        with open(dest_lbl_path, "w") as f: f.write("\n".join(label_lines))

        metadata = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, metadata)
        update_project_dataset_yaml(project_id)
        logger.info(f"Image '{data.original_filename}' ajoutée à projet '{project_id}' set '{data.target_set}'.")
        return metadata
    except Exception as e:
        logger.error(f"Erreur ajout image projet '{project_id}': {e}", exc_info=True)
        if dest_img_path.exists(): dest_img_path.unlink(missing_ok=True)
        if dest_lbl_path.exists(): dest_lbl_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur ajout image: {str(e)}")

@app.post("/finetune_projects/{project_id}/split_dataset", response_model=ProjectInfo, tags=["Projects Data"])
async def split_project_dataset(project_id: str, split_request: SplitDatasetRequest):
    metadata = load_project_metadata(project_id)
    if not metadata: raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    
    # Valider la somme des ratios
    if not np.isclose(split_request.train_ratio + split_request.val_ratio + split_request.test_ratio, 1.0):
        raise HTTPException(status_code=400, detail="La somme des ratios train, val, et test doit être égale à 1.0")

    if metadata.is_split:
        logger.info(f"Projet '{project_id}' déjà splitté. Nettoyage avant re-division...")
        for s_name in ["train", "val", "test"]:
            s_img_dir = get_project_images_dir(project_id, s_name)
            s_lbl_dir = get_project_labels_dir(project_id, s_name)
            target_root_img_dir = get_project_images_dir(project_id)
            target_root_lbl_dir = get_project_labels_dir(project_id)
            if s_img_dir.exists():
                for item in s_img_dir.iterdir(): 
                    if item.is_file(): shutil.move(str(item), str(target_root_img_dir / item.name))
            if s_lbl_dir.exists():
                for item in s_lbl_dir.iterdir(): 
                    if item.is_file(): shutil.move(str(item), str(target_root_lbl_dir / item.name))
        metadata.is_split = False

    source_images_dir = get_project_images_dir(project_id)
    source_labels_dir = get_project_labels_dir(project_id)
    all_image_files = [f for f in source_images_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp'] and not f.name.startswith('.')]
    if not all_image_files: raise HTTPException(status_code=400, detail="Aucune image à la racine de 'images/' pour division.")

    if split_request.shuffle: random.shuffle(all_image_files)
    num_total = len(all_image_files)
    num_train = int(split_request.train_ratio * num_total)
    num_val = min(int(split_request.val_ratio * num_total), num_total - num_train)
    num_test = num_total - num_train - num_val
    splits = {"train": all_image_files[:num_train], "val": all_image_files[num_train:num_train+num_val], "test": all_image_files[num_train+num_val:]}
    logger.info(f"Division projet {project_id}: Total={num_total}, Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

    try:
        for set_name, image_files_in_set in splits.items():
            dest_img_dir_set = get_project_images_dir(project_id, set_name); dest_img_dir_set.mkdir(parents=True, exist_ok=True)
            dest_lbl_dir_set = get_project_labels_dir(project_id, set_name); dest_lbl_dir_set.mkdir(parents=True, exist_ok=True)
            for img_file_path in image_files_in_set:
                shutil.move(str(img_file_path), str(dest_img_dir_set / img_file_path.name))
                label_file_name = f"{img_file_path.stem}.txt"
                source_label_path = source_labels_dir / label_file_name
                if source_label_path.exists(): shutil.move(str(source_label_path), str(dest_lbl_dir_set / label_file_name))
        metadata_updated = _update_project_image_counts_and_stats(project_id, metadata)
        save_project_metadata(project_id, metadata_updated)
        update_project_dataset_yaml(project_id)
        return metadata_updated
    except Exception as e:
        logger.error(f"Erreur division dataset projet '{project_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur division dataset: {str(e)}.")

@app.post("/finetune_projects/{project_id}/update_yaml", response_model=ProjectInfo, tags=["Projects Data"])
async def force_update_project_yaml(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    logger.info(f"Demande de mise à jour manuelle du dataset.yaml pour le projet {project_id}.")
    update_project_dataset_yaml(project_id)
    updated_metadata = load_project_metadata(project_id)
    if not updated_metadata:
         raise HTTPException(status_code=500, detail="Erreur rechargement métadonnées après màj YAML.")
    return updated_metadata

@app.post("/finetune_projects/{project_id}/export_coco", tags=["Projects Data"])
async def export_project_to_coco(project_id: str):
    metadata = load_project_metadata(project_id)
    if not metadata: raise HTTPException(status_code=404, detail=f"Projet '{project_id}' non trouvé.")
    
    # L'export COCO a plus de sens si le dataset est splitté car il combine train/val
    # Cependant, on peut aussi l'autoriser si non splitté, en utilisant toutes les images.
    # if not metadata.is_split and metadata.image_count > 0 :
    #      raise HTTPException(status_code=400, detail=f"Projet '{project_id}' n'a pas été splitté. Veuillez d'abord utiliser /split_dataset.")
    if metadata.image_count == 0 :
         raise HTTPException(status_code=400, detail=f"Projet '{project_id}' n'a pas d'images pour l'export COCO.")


    coco_output = {
        "info": {"description": f"COCO export: {metadata.name} ({project_id})", "version": "1.0", "year": datetime.now(timezone.utc).year, "date_created": datetime.now(timezone.utc).isoformat()},
        "licenses": [{"url": "about:blank", "id": 0, "name": "No License"}],
        "images": [], "annotations": [], "categories": []
    }
    for class_name, class_id in metadata.class_mapping.items():
        coco_output["categories"].append({"id": class_id, "name": class_name, "supercategory": "object"})
    
    annotation_id_counter = 1
    image_id_counter = 1
    
    sets_to_export = []
    if metadata.is_split:
        if metadata.train_image_count > 0: sets_to_export.append("train")
        if metadata.val_image_count > 0: sets_to_export.append("val")
    elif metadata.image_count > 0: # Non splitté, mais des images à la racine
        sets_to_export.append(None) # None indiquera la racine

    if not sets_to_export:
        raise HTTPException(status_code=400, detail="Aucun set (train/val ou racine) avec des images à exporter.")

    for subset_name_or_none in sets_to_export:
        image_dir_subset = get_project_images_dir(project_id, subset_name_or_none)
        label_dir_subset = get_project_labels_dir(project_id, subset_name_or_none)

        if not image_dir_subset.exists() or not label_dir_subset.exists():
            logger.info(f"Dossier images/labels pour set '{subset_name_or_none or 'racine'}' du projet '{project_id}' non trouvé. Ignoré.")
            continue

        for img_file in image_dir_subset.iterdir():
            if not (img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']): continue
            try:
                pil_img = Image.open(img_file); width, height = pil_img.size
            except Exception as e_img: logger.warning(f"Impossible de lire {img_file} pour export COCO: {e_img}"); continue

            current_image_id_coco = image_id_counter
            file_name_for_coco = f"{subset_name_or_none or 'root'}/{img_file.name}" if subset_name_or_none else img_file.name
            coco_output["images"].append({
                "id": current_image_id_coco, "width": width, "height": height, "file_name": file_name_for_coco,
                "license": 0, "date_captured": datetime.now(timezone.utc).isoformat()
            })
            image_id_counter += 1

            label_file = label_dir_subset / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f_label:
                    for line in f_label:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            class_id_yolo = int(parts[0])
                            x_c_n, y_c_n, w_n, h_n = map(float, parts[1:])
                            abs_w = w_n * width; abs_h = h_n * height
                            abs_x_min = (x_c_n * width) - (abs_w / 2); abs_y_min = (y_c_n * height) - (abs_h / 2)
                            coco_output["annotations"].append({
                                "id": annotation_id_counter, "image_id": current_image_id_coco,
                                "category_id": class_id_yolo, # YOLO class_id est déjà l'ID numérique
                                "bbox": [round(abs_x_min,2), round(abs_y_min,2), round(abs_w,2), round(abs_h,2)],
                                "area": round(abs_w * abs_h, 2), "iscrowd": 0, "segmentation": []
                            })
                            annotation_id_counter += 1
    
    if not coco_output["images"]:
        raise HTTPException(status_code=400, detail="Aucune image/annotation valide trouvée pour l'export COCO.")
    coco_filename = f"{metadata.name.replace(' ', '_')}_coco_export_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}.json"
    return JSONResponse(content=coco_output, media_type="application/json", headers={"Content-Disposition": f"attachment; filename={coco_filename}"})

# --- Fine-Tuning et Gestion des Runs ---
# ... (Le reste du code est supposé être identique à v1.0.1 et déjà corrigé pour les dates)
def _run_finetune_background_task(task: TaskStatus, project_id: str, base_model_path_str: str, ft_config: FineTuneRunConfig, run_name_override: Optional[str], save_period: int):
    _update_task_status(task.task_id, "running", "Préparation fine-tuning...", progress=0.01)
    try:
        project_metadata = load_project_metadata(project_id)
        if not project_metadata: raise ValueError(f"Métadonnées projet {project_id} non trouvées.")
        update_project_dataset_yaml(project_id)
        project_metadata = load_project_metadata(project_id) # Recharger
        dataset_yaml_path_str = project_metadata.dataset_yaml_path
        if not dataset_yaml_path_str or not Path(dataset_yaml_path_str).exists():
            raise ValueError(f"dataset.yaml pour projet {project_id} non trouvé.")
        dataset_yaml_path = Path(dataset_yaml_path_str)
        with open(dataset_yaml_path, 'r') as f_yaml: yaml_data = yaml.safe_load(f_yaml)
        
        if yaml_data.get('nc', 0) == 0 and project_metadata.image_count > 0:
            raise ValueError("Fine-tuning annulé: nc=0 dans dataset.yaml mais des images existent.")
        if not yaml_data.get('train') and project_metadata.image_count > 0 : # S'il y a des images, il faut un set train
             raise ValueError("Fine-tuning annulé: dataset.yaml ne spécifie pas de chemin 'train' alors que des images existent.")

        base_model_p = Path(base_model_path_str)
        if not base_model_p.is_absolute(): base_model_p = (HOME_DIR / base_model_path_str).resolve()
        if not base_model_p.is_file(): raise FileNotFoundError(f"Modèle base '{base_model_p}' non trouvé.")

        model_to_train = YOLO(str(base_model_p))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        project_name_slug = "".join(c if c.isalnum() else '_' for c in project_metadata.name.lower())[:20]
        effective_run_name = run_name_override if run_name_override else f"ft_{project_name_slug}_{base_model_p.stem}_{timestamp}"
        effective_run_name = "".join(c if c.isalnum() or c in ['_', '-'] else '_' for c in effective_run_name)
        output_dir_for_this_run = get_run_dir(effective_run_name)
        if output_dir_for_this_run.exists() and any(output_dir_for_this_run.iterdir()):
            effective_run_name = f"{effective_run_name}_{uuid.uuid4().hex[:4]}"
            output_dir_for_this_run = get_run_dir(effective_run_name)
            logger.warning(f"Nom de run existait. Nouveau: '{effective_run_name}'")

        _update_task_status(task.task_id, "running", f"Entraînement sur {ft_config.device}. Run: {effective_run_name}", progress=0.05)
        train_args = ft_config.model_dump(exclude={"augmentations"})
        train_args.update(ft_config.augmentations.model_dump())
        train_args.update({
            "data": str(dataset_yaml_path), "project": str(FINETUNED_MODELS_OUTPUT_DIR),
            "name": effective_run_name, "exist_ok": True, "save_period": save_period,
            "workers": max(1, os.cpu_count() // 2 - 1 if os.cpu_count() else 1),
            # AJOUT D'UN CALLBACK POUR LA PROGRESSION
            "callbacks": {
                "on_epoch_end": lambda trainer: _update_task_status(task.task_id, "running", f"Epoch {trainer.epoch+1}/{trainer.epochs} terminée.", progress= (trainer.epoch+1)/trainer.epochs * 0.9 + 0.05) # 0.05 à 0.95 pour training
            }
        })
        logger.info(f"Tâche FT [{task.task_id}]: Args train: {train_args}")
        results = model_to_train.train(**train_args)
        
        actual_run_output_dir = Path(results.save_dir)
        final_run_name = actual_run_output_dir.name if actual_run_output_dir.name != effective_run_name else effective_run_name
        final_output_dir_for_run = FINETUNED_MODELS_OUTPUT_DIR / final_run_name # Assurer le bon chemin parent

        best_model_path = final_output_dir_for_run / "weights" / "best.pt"
        last_model_path = final_output_dir_for_run / "weights" / "last.pt"
        model_name_pt_reported, final_model_to_report_path = ("best.pt", best_model_path) if best_model_path.exists() else \
                                                             ("last.pt", last_model_path) if last_model_path.exists() else \
                                                             (None, None)
        if not final_model_to_report_path: raise FileNotFoundError(f"best.pt ou last.pt non trouvés dans {final_output_dir_for_run / 'weights'}.")

        artifacts_list = []
        for pattern in ["results.png", "confusion_matrix.png", "labels*.jpg", "*.csv", "args.yaml", "hyp.yaml"]:
            for artifact_file in final_output_dir_for_run.glob(pattern):
                if artifact_file.is_file():
                    file_type = "image" if artifact_file.suffix.lower() in ['.png', '.jpg'] else \
                                "csv" if artifact_file.suffix.lower() == '.csv' else \
                                "config" if artifact_file.suffix.lower() == '.yaml' else "log"
                    artifacts_list.append(RunArtifact(name=artifact_file.name, path_on_server_relative_to_run_dir=str(artifact_file.relative_to(final_output_dir_for_run)), type=file_type))
        tb_events = list(final_output_dir_for_run.glob("events.out.tfevents.*"))
        if tb_events: artifacts_list.append(RunArtifact(name="tensorboard_logs", path_on_server_relative_to_run_dir=str(tb_events[0].relative_to(final_output_dir_for_run)), type="tensorboard_logs"))

        final_metrics_dict = results.results_dict if hasattr(results, 'results_dict') else {}
        best_fitness_epoch = results.epoch + 1 if hasattr(results, 'epoch') and results.epoch is not None else None
        run_metadata = FineTunedModelInfo(
            run_name=final_run_name, model_name_pt=model_name_pt_reported,
            model_path_abs=str(final_model_to_report_path.resolve()),
            model_path_relative_to_home=str(final_model_to_report_path.relative_to(HOME_DIR)),
            project_id_source=project_id, base_model_used=base_model_p.name,
            training_date=datetime.now(timezone.utc), run_config_used=ft_config.model_dump(),
            final_metrics=final_metrics_dict, dataset_yaml_used_path_abs=str(dataset_yaml_path.resolve()),
            artifacts=artifacts_list, output_dir_abs=str(final_output_dir_for_run.resolve()),
            output_dir_relative_to_home=str(final_output_dir_for_run.relative_to(HOME_DIR)),
            best_epoch=best_fitness_epoch,
        )
        with open(get_run_metadata_path(final_run_name), "w") as f_meta: json.dump(run_metadata.model_dump(mode='json'), f_meta, indent=4)
        _update_task_status(task.task_id, "completed", f"Terminé. Modèle: {run_metadata.model_path_relative_to_home}", result=run_metadata.model_dump(mode='json'), progress=1.0)
        logger.info(f"Tâche FT [{task.task_id}] terminée: {final_run_name}")
    except Exception as e:
        logger.error(f"Erreur tâche FT [{task.task_id}]: {e}", exc_info=True)
        _update_task_status(task.task_id, "error", f"Erreur: {str(e)}", progress=task.progress or 0.0) # Garder la progression si déjà définie


# --- Endpoints de Fine-Tuning, Tâches, Modèles Fine-Tunés, Export, HPO, LLM ---
# ... (La plupart sont identiques à la version précédente, sauf s'il y a des dépendances aux corrections ci-dessus)
# S'assurer que les appels à load_tasks_db et load_project_metadata utilisent les versions corrigées.

@app.post("/finetune", response_model=TaskStatusResponse, tags=["Fine-Tuning"])
async def start_finetune_model_on_project(request_data: FineTuneRequest, background_tasks: BackgroundTasks):
    if not (get_project_dir(request_data.project_id)).exists():
        raise HTTPException(status_code=404, detail=f"Projet '{request_data.project_id}' non trouvé.")
    prospective_base_model_path = Path(request_data.base_model_path)
    path_to_use_for_task_str = ""
    if prospective_base_model_path.is_absolute() and prospective_base_model_path.is_file():
        path_to_use_for_task_str = str(prospective_base_model_path.resolve())
    else:
        path_rel_to_home = (HOME_DIR / request_data.base_model_path).resolve()
        if path_rel_to_home.is_file(): path_to_use_for_task_str = str(path_rel_to_home)
        else:
             simple_name_path = (HOME_DIR / Path(request_data.base_model_path).name).resolve()
             if simple_name_path.is_file(): path_to_use_for_task_str = str(simple_name_path)
             else: raise HTTPException(status_code=404, detail=f"Modèle base '{request_data.base_model_path}' non trouvé.")
    
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type="finetune", message="Tâche fine-tuning en file d'attente.")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(
        _run_finetune_background_task, new_task, request_data.project_id,
        path_to_use_for_task_str, request_data.run_config, request_data.run_name, request_data.save_period
    )
    logger.info(f"Tâche FT [{new_task.task_id}] projet '{request_data.project_id}' ajoutée.")
    return TaskStatusResponse(**new_task.model_dump())

@app.get("/finetune_tasks", response_model=List[TaskStatusResponse], tags=["Fine-Tuning Tasks"])
async def list_all_tasks():
    tasks_db = load_tasks_db()
    return sorted([TaskStatusResponse(**task.model_dump()) for task in tasks_db.values()], key=lambda t: t.created_at, reverse=True)

@app.get("/finetune_tasks/{task_id}", response_model=TaskStatusResponse, tags=["Fine-Tuning Tasks"])
async def get_task_status(task_id: str):
    tasks_db = load_tasks_db()
    task = tasks_db.get(task_id)
    if not task: raise HTTPException(status_code=404, detail=f"Tâche ID '{task_id}' non trouvée.")
    return TaskStatusResponse(**task.model_dump())

@app.get("/finetuned_models", response_model=List[FineTunedModelInfo], tags=["Fine-Tuned Models"])
async def list_finetuned_models():
    tuned_models_info = []
    if not FINETUNED_MODELS_OUTPUT_DIR.exists(): return []
    for run_dir_item in FINETUNED_MODELS_OUTPUT_DIR.iterdir():
        if run_dir_item.is_dir():
            run_name = run_dir_item.name
            meta_file_path = get_run_metadata_path(run_name)
            if meta_file_path.exists():
                try:
                    with open(meta_file_path, 'r') as f_meta: run_meta_data = json.load(f_meta)
                    # Correction pour dates dans run_meta_data si nécessaire avant Pydantic
                    if isinstance(run_meta_data.get("training_date"), str):
                        td_str = run_meta_data["training_date"]
                        if td_str.endswith('Z'): td_str = td_str[:-1] + '+00:00'
                        run_meta_data["training_date"] = datetime.fromisoformat(td_str)

                    info = FineTunedModelInfo(**run_meta_data)
                    if not Path(info.model_path_abs).exists():
                        logger.warning(f"Modèle .pt '{info.model_path_abs}' pour run '{run_name}' non existant. Run ignoré.")
                        continue
                    tuned_models_info.append(info)
                except Exception as e:
                    logger.warning(f"Erreur lecture/validation métadonnées run '{run_name}': {e}. Run ignoré.")
    return sorted(tuned_models_info, key=lambda m: m.training_date, reverse=True)

@app.get("/finetuned_models/{run_name}", response_model=FineTunedModelInfo, tags=["Fine-Tuned Models"])
async def get_finetuned_model_details(run_name: str):
    meta_path = get_run_metadata_path(run_name)
    if not meta_path.exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    try:
        with open(meta_path, 'r') as f: data = json.load(f)
        if isinstance(data.get("training_date"), str): # Correction date
            td_str = data["training_date"]
            if td_str.endswith('Z'): td_str = td_str[:-1] + '+00:00'
            data["training_date"] = datetime.fromisoformat(td_str)
        return FineTunedModelInfo(**data)
    except Exception as e:
        logger.error(f"Erreur chargement métadonnées run {run_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lecture métadonnées run: {str(e)}")

@app.put("/finetuned_models/{run_name}/notes", response_model=FineTunedModelInfo, tags=["Fine-Tuned Models"])
async def update_finetuned_model_notes(run_name: str, notes: str = Body(..., embed=True, max_length=1000)):
    meta_path = get_run_metadata_path(run_name)
    if not meta_path.exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    try:
        with open(meta_path, 'r') as f: data = json.load(f)
        if isinstance(data.get("training_date"), str): # Correction date
            td_str = data["training_date"]
            if td_str.endswith('Z'): td_str = td_str[:-1] + '+00:00'
            data["training_date"] = datetime.fromisoformat(td_str)
        model_info = FineTunedModelInfo(**data)
        model_info.notes = notes
        with open(meta_path, 'w') as f: json.dump(model_info.model_dump(mode='json'), f, indent=4)
        logger.info(f"Notes màj pour run '{run_name}'.")
        return model_info
    except Exception as e:
        logger.error(f"Erreur màj notes run {run_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur màj notes: {str(e)}")

# Les endpoints /finetuned_models/{run_name}/artifacts/{artifact_filename:path}, /metrics_details, /download_model/{model_path_b64}
# et les sections Export et HPO sont supposés être corrects et peuvent rester comme dans la version précédente.
# Je vais les inclure pour la complétude.

@app.get("/finetuned_models/{run_name}/artifacts/{artifact_filename:path}", tags=["Fine-Tuned Models"])
async def get_run_artifact(run_name: str, artifact_filename: str):
    run_dir = get_run_dir(run_name)
    if not run_dir.is_dir(): raise HTTPException(status_code=404, detail=f"Dossier run '{run_name}' non trouvé.")
    artifact_path = (run_dir / artifact_filename).resolve()
    if not str(artifact_path).startswith(str(run_dir.resolve())) or not artifact_path.is_file():
        raise HTTPException(status_code=404, detail=f"Artefact '{artifact_filename}' non trouvé/autorisé.")
    return FileResponse(path=artifact_path, filename=artifact_path.name)

@app.get("/finetuned_models/{run_name}/metrics_details", tags=["Fine-Tuned Models"])
async def get_run_metrics_details_from_csv(run_name: str):
    results_csv_path = get_run_dir(run_name) / "results.csv"
    if not results_csv_path.exists(): raise HTTPException(status_code=404, detail=f"results.csv non trouvé pour run '{run_name}'.")
    metrics_data = []
    try:
        with open(results_csv_path, 'r', newline='') as cf: reader = csv.DictReader(cf)
        for row in reader: metrics_data.append({k.strip(): (float(v) if v.replace('.', '', 1).isdigit() else v) for k,v in row.items()})
        return metrics_data
    except Exception as e:
        logger.error(f"Erreur lecture results.csv run {run_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur lecture métriques: {str(e)}")

@app.get("/download_model/{model_path_b64}", tags=["Models"])
async def download_any_model_by_b64_path(model_path_b64: str):
    try:
        relative_model_path_str = base64.urlsafe_b64decode(model_path_b64.encode('utf-8') + b'===').decode('utf-8')
    except Exception: raise HTTPException(status_code=400, detail="Chemin modèle encodé invalide.")
    model_file_path = (HOME_DIR / Path(relative_model_path_str)).resolve()
    if not model_file_path.is_file() or not str(model_file_path).startswith(str(HOME_DIR.resolve())):
        raise HTTPException(status_code=404, detail="Fichier modèle non trouvé ou accès non autorisé.")
    is_base = str(model_file_path.parent.resolve()) == str(MODELS_DIR.resolve())
    is_ft = str(FINETUNED_MODELS_OUTPUT_DIR.resolve()) in str(model_file_path.resolve().parents)
    if not (is_base or is_ft): raise HTTPException(status_code=403, detail="Chemin non autorisé.")
    return FileResponse(path=model_file_path, filename=model_file_path.name, media_type='application/octet-stream')

# --- Export de Modèles (ONNX, etc.) ---
# _run_export_model_task et /finetuned_models/{run_name}/export restent identiques
def _run_export_model_task(task: TaskStatus, run_name: str, export_config: ExportFormat):
    _update_task_status(task.task_id, "running", f"Préparation export {run_name} vers {export_config.format}...", progress=0.01)
    try:
        run_meta_path = get_run_metadata_path(run_name)
        if not run_meta_path.exists(): raise FileNotFoundError(f"Métadonnées run '{run_name}' non trouvées.")
        with open(run_meta_path, 'r') as f_meta: run_info_data = json.load(f_meta) # Charger comme dict d'abord
        # Correction de date pour Pydantic
        if isinstance(run_info_data.get("training_date"), str):
            td_str = run_info_data["training_date"]
            if td_str.endswith('Z'): td_str = td_str[:-1] + '+00:00'
            run_info_data["training_date"] = datetime.fromisoformat(td_str)
        run_info = FineTunedModelInfo(**run_info_data)

        source_model_pt_path = Path(run_info.model_path_abs)
        if not source_model_pt_path.exists(): raise FileNotFoundError(f"Modèle .pt '{source_model_pt_path}' non trouvé.")
        
        model_to_export = YOLO(str(source_model_pt_path))
        export_params = export_config.model_dump(exclude_none=True)
        if export_params.get("imgsz") is None and run_info.run_config_used.get("img_size"):
            export_params["imgsz"] = run_info.run_config_used["img_size"]
        
        _update_task_status(task.task_id, "running", f"Exportation vers {export_config.format}...", progress=0.3)
        exported_file_path_str = model_to_export.export(**export_params)
        exported_file_path = Path(exported_file_path_str)
        if not exported_file_path.exists(): raise FileNotFoundError(f"Fichier exporté {export_config.format} non trouvé à {exported_file_path_str}")

        new_artifact = RunArtifact(
            name=exported_file_path.name,
            path_on_server_relative_to_run_dir=str(exported_file_path.relative_to(get_run_dir(run_name))),
            type=export_config.format
        )
        if not any(art.name == new_artifact.name and art.type == new_artifact.type for art in run_info.artifacts):
            run_info.artifacts.append(new_artifact)
            with open(run_meta_path, "w") as f_meta_update: json.dump(run_info.model_dump(mode='json'), f_meta_update, indent=4)
        
        result_payload = {
            "exported_model_name": exported_file_path.name, "exported_model_path_abs": str(exported_file_path.resolve()),
            "exported_model_path_relative_to_home": str(exported_file_path.relative_to(HOME_DIR)),
            "format": export_config.format, "run_name_source": run_name
        }
        _update_task_status(task.task_id, "completed", f"Export terminé: {exported_file_path.name}", result=result_payload, progress=1.0)
        logger.info(f"Tâche Export [{task.task_id}] terminée: {exported_file_path}")
    except Exception as e:
        logger.error(f"Erreur export run '{run_name}': {e}", exc_info=True)
        _update_task_status(task.task_id, "error", f"Erreur export: {str(e)}")

@app.post("/finetuned_models/{run_name}/export", response_model=TaskStatusResponse, tags=["Fine-Tuned Models Export"])
async def export_finetuned_model(run_name: str, export_config: ExportFormat, background_tasks: BackgroundTasks):
    if not get_run_dir(run_name).exists(): raise HTTPException(status_code=404, detail=f"Run '{run_name}' non trouvé.")
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type=f"export_{export_config.format}", message=f"Tâche export {export_config.format} pour run '{run_name}' en attente.")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(_run_export_model_task, new_task, run_name, export_config)
    return TaskStatusResponse(**new_task.model_dump())

# --- Hyperparameter Optimization (HPO) ---
# _run_hpo_task et /hpo_tune restent identiques
def _run_hpo_task(task: TaskStatus, hpo_request: HyperparameterTuneRequest):
    _update_task_status(task.task_id, "running", f"Préparation HPO projet {hpo_request.project_id}...", progress=0.01)
    try:
        project_metadata = load_project_metadata(hpo_request.project_id)
        if not project_metadata: raise ValueError(f"Métadonnées projet {hpo_request.project_id} non trouvées pour HPO.")
        update_project_dataset_yaml(hpo_request.project_id)
        project_metadata = load_project_metadata(hpo_request.project_id)
        dataset_yaml_path_str = project_metadata.dataset_yaml_path
        if not dataset_yaml_path_str or not Path(dataset_yaml_path_str).exists():
            raise ValueError(f"dataset.yaml projet {hpo_request.project_id} non trouvé pour HPO.")

        base_model_p = Path(hpo_request.base_model_path)
        if not base_model_p.is_absolute(): base_model_p = (HOME_DIR / hpo_request.base_model_path).resolve()
        if not base_model_p.is_file(): raise FileNotFoundError(f"Modèle base '{base_model_p}' non trouvé pour HPO.")
        
        model_for_hpo = YOLO(str(base_model_p))
        base_tune_config = hpo_request.base_run_config.model_dump(exclude={"augmentations"})
        base_tune_config.update(hpo_request.base_run_config.augmentations.model_dump())
        
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        hpo_run_name = f"hpo_{project_metadata.name.replace(' ','_')}_{timestamp}"
        tune_call_args = {
            "data": dataset_yaml_path_str, "epochs": hpo_request.base_run_config.epochs,
            "iterations": hpo_request.num_trials, "optimizer": hpo_request.base_run_config.optimizer,
            "project": str(FINETUNED_MODELS_OUTPUT_DIR), "name": hpo_run_name,
            "imgsz": hpo_request.base_run_config.img_size, "batch": hpo_request.base_run_config.batch_size,
            "device": hpo_request.base_run_config.device, "val": True,
        }
        tune_call_args.update(base_tune_config) # Appliquer la config de base
        _update_task_status(task.task_id, "running", f"HPO en cours ({hpo_request.num_trials} essais)...", progress=0.1)
        best_model_after_tune_path_str = model_for_hpo.tune(**tune_call_args)
        best_model_path = Path(best_model_after_tune_path_str)
        if not best_model_path.exists() or not best_model_path.name == "best.pt":
            raise FileNotFoundError(f"Meilleur modèle non trouvé après HPO. Obtenu: {best_model_after_tune_path_str}")

        hpo_output_dir = get_run_dir(hpo_run_name)
        best_hyp_yaml_path = hpo_output_dir / "hyp.yaml"
        best_run_config_used = {}
        if best_hyp_yaml_path.exists():
            with open(best_hyp_yaml_path, 'r') as f_hyp: best_run_config_used = yaml.safe_load(f_hyp)
        
        hpo_artifacts = [] # Collecter artefacts
        for pattern in ["*.csv", "*.png", "*.log", "hyp.yaml", "args.yaml"]:
            for artifact_file in hpo_output_dir.glob(pattern):
                 if artifact_file.is_file():
                    file_type = "image" if artifact_file.suffix.lower() in ['.png', '.jpg'] else "csv" if artifact_file.suffix.lower() == '.csv' else "config" if artifact_file.suffix.lower() == '.yaml' else "log"
                    hpo_artifacts.append(RunArtifact(name=artifact_file.name, path_on_server_relative_to_run_dir=str(artifact_file.relative_to(hpo_output_dir)), type=file_type))
        
        hpo_run_metadata = FineTunedModelInfo(
            run_name=hpo_run_name, model_name_pt="best.pt", model_path_abs=str(best_model_path.resolve()),
            model_path_relative_to_home=str(best_model_path.relative_to(HOME_DIR)),
            project_id_source=hpo_request.project_id, base_model_used=base_model_p.name,
            training_date=datetime.now(timezone.utc), run_config_used=best_run_config_used,
            dataset_yaml_used_path_abs=dataset_yaml_path_str, artifacts=hpo_artifacts,
            output_dir_abs=str(hpo_output_dir.resolve()), output_dir_relative_to_home=str(hpo_output_dir.relative_to(HOME_DIR)),
            notes=f"Résultat HPO avec {hpo_request.num_trials} essais."
        )
        with open(get_run_metadata_path(hpo_run_name), "w") as f_hpo_meta: json.dump(hpo_run_metadata.model_dump(mode='json'), f_hpo_meta, indent=4)
        _update_task_status(task.task_id, "completed", f"HPO terminé. Modèle: {best_model_path.name} dans {hpo_run_name}", result=hpo_run_metadata.model_dump(mode='json'), progress=1.0)
    except Exception as e:
        logger.error(f"Erreur HPO: {e}", exc_info=True)
        _update_task_status(task.task_id, "error", f"Erreur HPO: {str(e)}")

@app.post("/hpo_tune", response_model=TaskStatusResponse, tags=["Hyperparameter Optimization"])
async def start_hyperparameter_optimization(hpo_request: HyperparameterTuneRequest, background_tasks: BackgroundTasks):
    if not get_project_dir(hpo_request.project_id).exists(): raise HTTPException(status_code=404, detail=f"Projet '{hpo_request.project_id}' non trouvé pour HPO.")
    tasks_db = load_tasks_db()
    new_task = _create_task_entry(task_type="hpo_tune", message=f"Tâche HPO projet '{hpo_request.project_id}' en attente.")
    tasks_db[new_task.task_id] = new_task
    save_tasks_db(tasks_db)
    background_tasks.add_task(_run_hpo_task, new_task, hpo_request)
    return TaskStatusResponse(**new_task.model_dump())


# --- Endpoint LLM ---
# ... (Identique à la version précédente)
@app.post("/describe_llm", response_model=LLMDescribeResponse, tags=["LLM"])
async def describe_image_with_llm(request: LLMDescribeRequest):
    if not EXTERNAL_LLM_FULL_URL or ("your-llm-api-ngrok-url" in EXTERNAL_NGROK_LLM_BASE_URL and EXTERNAL_NGROK_LLM_BASE_URL != "https://80f2-193-55-66-240.ngrok-free.app"):
        if "your-llm-api-ngrok-url" in EXTERNAL_NGROK_LLM_BASE_URL :
            logger.warning("URL LLM externe non configurée ou utilise la valeur par défaut.")
            raise HTTPException(status_code=503, detail="Service LLM non configuré correctement.")
    try:
        payload = {"image_base64": request.image_base64, "prompt": request.prompt}
        headers = {"Content-Type": "application/json", "ngrok-skip-browser-warning": "true"}
        logger.debug(f"Envoi de la requête LLM à {EXTERNAL_LLM_FULL_URL}...")
        response = requests.post(EXTERNAL_LLM_FULL_URL, json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        response_data = response.json(); description = None
        keys_to_check = ["description", "response", "text", "content", "answer", "generated_text"]
        for key in keys_to_check:
            if key in response_data and isinstance(response_data[key], str): description = response_data[key]; break
        if description is None and "choices" in response_data and isinstance(response_data["choices"], list) and response_data["choices"]:
            first_choice = response_data["choices"][0]
            if "message" in first_choice and "content" in first_choice["message"] and isinstance(first_choice["message"]["content"], str):
                description = first_choice["message"]["content"]
            elif "text" in first_choice and isinstance(first_choice["text"], str): description = first_choice["text"]
        desc_text = description.strip() if isinstance(description, str) else None
        if not desc_text:
            logger.warning(f"Format réponse LLM non reconnu. Réponse (partielle): {json.dumps(response_data)[:300]}")
            desc_text = "Format de réponse LLM non reconnu. Détails logs serveur."
        else: logger.info(f"Description LLM reçue (préfixe): {desc_text[:100]}...")
        return LLMDescribeResponse(description=desc_text)
    except requests.exceptions.Timeout:
        logger.error("Timeout API LLM.")
        raise HTTPException(status_code=504, detail="API LLM timeout.")
    except requests.exceptions.HTTPError as http_err:
        error_content = "N/A"; status_code_from_llm = 503
        if http_err.response is not None:
            status_code_from_llm = http_err.response.status_code
            try: error_content_json = http_err.response.json(); error_content = error_content_json.get("detail") or error_content_json.get("error") or str(error_content_json)[:200]
            except json.JSONDecodeError: error_content = http_err.response.text[:200]
        logger.error(f"Erreur HTTP {status_code_from_llm} API LLM: {error_content}", exc_info=True)
        raise HTTPException(status_code=status_code_from_llm, detail=f"Erreur API LLM ({status_code_from_llm}): {error_content}")
    except Exception as e:
        logger.error(f"Erreur inattendue appel LLM: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur serveur (LLM): {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    host_ip = os.environ.get("HOST", "0.0.0.0")
    logger.info(f"Démarrage du serveur FastAPI (Advanced ML Suite v1.0.2) sur http://{host_ip}:{port}")
    uvicorn.run(app, host=host_ip, port=port)