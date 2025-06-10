# YOLO Annotator Pro - Advanced ML Suite


YOLO Annotator Pro est une application web complète conçue pour faciliter le cycle de vie des projets de détection d'objets avec les modèles YOLO (Ultralytics). Elle offre une interface utilisateur pour l'annotation d'images, la prédiction, la gestion de datasets, le fine-tuning de modèles, l'optimisation d'hyperparamètres (HPO), et plus encore.

L'application se compose de :
1.  **Backend FastAPI (`detectngrok.py`)**: Gère toute la logique métier, l'interaction avec les modèles YOLO, la gestion des fichiers et les tâches asynchrones.
2.  **Frontend HTML/CSS/JS (`index2.html`)**: Fournit une interface utilisateur interactive dans le navigateur.

---

## Table des Matières

*   [Fonctionnalités Principales](#fonctionnalités-principales)
*   [Prérequis](#prérequis)
*   [Installation](#installation)
    *   [Cloner le dépôt](#1-cloner-le-dépôt)
    *   [Environnement Virtuel](#2-créer-et-activer-un-environnement-virtuel)
    *   [Dépendances](#3-installer-les-dépendances)
    *   [Configuration Initiale](#4-configuration-initiale-importante)
*   [Lancer l'Application](#lancer-lapplication)
    *   [Démarrer le Backend](#1-démarrer-le-serveur-backend-fastapi)
    *   [Configurer NGROK (Optionnel pour accès externe)](#2-configurer-ngrok-optionnel-pour-accès-externe)
    *   [Accéder à l'Interface Utilisateur](#3-accéder-à-linterface-utilisateur)
*   [Utilisation](#utilisation)
*   [Structure des Dossiers](#structure-des-dossiers-côté-serveur-sous-home_dir)
*   [Contribution](#contribution)
*   [Licence](#licence)

---

## Fonctionnalités Principales

*   **Annotation d'Images**: Dessiner des boîtes englobantes (bounding boxes) et assigner des labels.
*   **Prédiction**: Utiliser des modèles YOLO pré-entraînés ou fine-tunés pour la détection d'objets. Support de SAHI pour la détection sur de grandes images.
*   **Description d'Image par LLM**: (Optionnel) S'interfacer avec un service LLM externe.
*   **Gestion de Projets de Fine-Tuning**:
    *   Créer, organiser, et supprimer des projets.
    *   Ajouter des images et annotations manuellement ou **importer des datasets au format ZIP** (avec structures `images/train`, `labels/train`, etc., ou détection de classes).
    *   Diviser les datasets (train/val/test), exporter en COCO, visualiser les statistiques.
*   **Fine-Tuning de Modèles YOLO**:
    *   Lancer des entraînements en arrière-plan avec configuration des hyperparamètres et augmentations.
*   **Gestion des Modèles Entraînés**:
    *   Lister modèles, métriques, artefacts. Exporter en ONNX, TorchScript, TensorRT. Télécharger les poids (`.pt`).
*   **Optimisation d'Hyperparamètres (HPO)**: Lancer des sessions de "tuning" YOLO.
*   **Suivi des Tâches**: Interface pour visualiser le statut des tâches longues.

## Prérequis

*   Python 3.8+
*   `pip` et `virtualenv` (recommandé)
*   Un GPU NVIDIA avec CUDA et cuDNN installés (fortement recommandé).
*   **(Optionnel)** Un compte [NGROK](https://ngrok.com/) si vous souhaitez :
    *   Exposer votre backend FastAPI sur internet.
    *   Utiliser la fonction LLM avec un service externe qui est lui-même exposé via NGROK.

## Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/Altay55stage/annotator-pro.git
cd annotator-pro

2. Créer et activer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END
3. Installer les dépendances

(Assurez-vous que le fichier requirements.txt est présent à la racine du projet)

pip install -r requirements.txt
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

Note: L'installation de torch peut nécessiter une commande spécifique en fonction de votre version de CUDA. Consultez pytorch.org pour la commande adaptée à votre système.

4. Configuration Initiale Importante

A. Modifier HOME_DIR dans detectngrok.py:
Ce chemin est CRUCIAL. Il définit où toutes les données de l'application (modèles, projets, uploads temporaires) seront stockées.
Ouvrez detectngrok.py et modifiez la ligne :

# Dans detectngrok.py
# HOME_DIR = Path("/home/acevik").resolve() # Exemple original
HOME_DIR = Path("/votre/chemin/personnel/vers/yolo_annotator_data").resolve() # MODIFIEZ CECI !
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END

Assurez-vous que ce répertoire existe ou que le script a les droits pour le créer, ainsi que pour y écrire. Les sous-dossiers nécessaires seront créés automatiquement.

B. (Optionnel) Placer les Modèles YOLO de Base:
Téléchargez des modèles YOLO pré-entraînés (ex: yolov8n.pt) depuis Ultralytics GitHub. Placez-les dans le répertoire que vous avez défini comme MODELS_DIR (par défaut, c'est le même que HOME_DIR).

C. (Optionnel) Configurer l'API LLM Externe:
Si vous souhaitez utiliser la fonctionnalité de description d'image par LLM avec un service externe, mettez à jour EXTERNAL_NGROK_LLM_BASE_URL dans detectngrok.py :

# Dans detectngrok.py
EXTERNAL_NGROK_LLM_BASE_URL = "https://VOTRE_URL_LLM.ngrok-free.app" # Ou une autre URL directe
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Python
IGNORE_WHEN_COPYING_END
Lancer l'Application
1. Démarrer le serveur backend FastAPI

Assurez-vous que votre environnement virtuel est activé.

uvicorn detectngrok:app --host 0.0.0.0 --port 8001 --reload
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

--host 0.0.0.0 rend le serveur accessible depuis d'autres machines sur votre réseau local.

--port 8001 est le port d'écoute. Changez-le si 8001 est déjà utilisé.

--reload est utile pour le développement (le serveur redémarre si les fichiers Python changent).

Le backend sera accessible localement à http://localhost:8001.

2. Configurer NGROK (Optionnel pour accès externe)

Si vous souhaitez accéder à votre application depuis l'extérieur de votre réseau local (par exemple, pour partager une démo ou si votre API LLM est externe et votre frontend doit y accéder via une URL publique), vous pouvez utiliser NGROK.

A. Installer NGROK: Suivez les instructions sur ngrok.com/download. Après le téléchargement, décompressez-le.

B. Configurer votre Authtoken NGROK:
Créez un compte gratuit sur dashboard.ngrok.com pour obtenir votre authtoken. Puis, exécutez (une seule fois) :

./ngrok config add-authtoken VOTRE_AUTHTOKEN
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Adaptez ./ngrok pour ngrok.exe sur Windows si nécessaire)

C. Exposer votre backend FastAPI (qui tourne sur le port 8001):
Dans un nouveau terminal, naviguez vers le dossier où vous avez décompressé NGROK et exécutez :

./ngrok http 8001
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
Bash
IGNORE_WHEN_COPYING_END

(Adaptez ./ngrok pour ngrok.exe sur Windows si nécessaire)
NGROK affichera une URL publique (ex: https://random-string.ngrok-free.app). C'est cette URL que vous utiliserez pour API_BASE_URL dans le frontend si vous accédez via NGROK.

Note : L'URL gratuite de NGROK change à chaque redémarrage de NGROK.

3. Accéder à l'Interface Utilisateur

Ouvrez le fichier index2.html directement dans votre navigateur web (ex: double-cliquez dessus ou file:///chemin/vers/le/projet/annotator-pro/index2.html).

IMPORTANT : Configurer API_BASE_URL dans index2.html:
Ouvrez index2.html dans un éditeur de texte et trouvez la ligne (vers le début de la balise <script>):

const API_BASE_URL = 'http://localhost:8001'; // Valeur par défaut pour accès local
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
JavaScript
IGNORE_WHEN_COPYING_END

Modifiez cette valeur en fonction de comment vous accédez au backend :

Accès local: const API_BASE_URL = 'http://localhost:8001'; (ou le port que vous avez utilisé).

Accès via NGROK: const API_BASE_URL = 'https://VOTRE_URL_NGROK.ngrok-free.app'; (utilisez l'URL fournie par NGROK).

Une fois index2.html configuré et ouvert, vous devriez voir l'interface et le statut API "Connecté".

Utilisation

L'interface utilisateur est divisée en plusieurs sections accessibles via le menu latéral :

Annoter / Prédire:

Chargez une image.

Chargez un modèle (.pt) depuis le serveur.

Cliquez sur "Analyser Image" pour obtenir des prédictions.

Activez le "Mode Dessin BBox" pour annoter manuellement.

Utilisez "Décrire (LLM)" (si configuré).

Ajoutez l'image annotée à un projet.

Projets:

Créez/gérez vos projets de fine-tuning.

Visualisez les détails, statistiques.

Modifiez, divisez le dataset, exportez en COCO, supprimez.

Importez un dataset ZIP : structure images/train/*.jpg, labels/train/*.txt, ou laissez le script détecter les classes et la structure.

Modèles Entraînés:

Lancez un nouveau fine-tuning.

Consultez les runs terminés, leurs détails, artefacts.

Exportez les modèles (.pt) en ONNX, etc., ou téléchargez-les.

Optimisation Hyperparams:

Lancez une recherche d'hyperparamètres (HPO). Les résultats apparaîtront comme des runs normaux.

Tâches en Cours:

Suivez la progression des tâches de fond (fine-tuning, HPO, export).

Journal Applicatif:

Logs du frontend pour le débogage.

Structure des Dossiers (côté serveur, sous HOME_DIR)

MODELS_DIR (par défaut HOME_DIR): Stockez ici vos modèles .pt de base (ex: yolov8n.pt).

TEMP_UPLOAD_DIR (temp_uploads_annotator): Fichiers temporaires.

FINETUNE_PROJECTS_DIR (finetune_projects_annotator): Données des projets (images, labels, dataset.yaml).

FINETUNED_MODELS_OUTPUT_DIR (finetuned_model_runs_annotator): Résultats des entraînements (runs).

TASKS_DB_FILE (annotator_tasks_db.json): Base de données JSON pour le statut des tâches.

Contribution

Les contributions sont les bienvenues ! N'hésitez pas à forker le dépôt, créer une branche, faire vos modifications et ouvrir une Pull Request.
