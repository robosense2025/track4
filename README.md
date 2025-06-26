# 🚁 RoboSense Track 4: Cross-Modal Drone Navigation

<div align="center">

**Official Baseline Implementation for Track 4**

*Based on GeoText-1652: Towards Natural Language-Guided Drones with Spatial Relation Matching*<br>(https://github.com/MultimodalGeo/GeoText-1652)

[![RoboSense Challenge](https://img.shields.io/badge/RoboSense-2025-blue)](https://robosense2025.github.io/)
[![Track 4](https://img.shields.io/badge/Track-Cross--Modal%20Drone%20Navigation-green)](https://robosense2025.github.io/track4)
[![IROS 2025](https://img.shields.io/badge/IROS-2025-red)](https://iros2025.org/)
[![CodaBench](https://img.shields.io/badge/CodaBench-Submit-purple)](https://www.codabench.org/competitions/9219/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

**🏆 Prize Pool: $2,000 USD for Track 4 Winners**

<p align="center">
  <img src="docs/figures/track4.jpg" align="center" width="60%">
</p>

</div>

## Challenge Overview

**Track 4: Cross-Modal Drone Navigation** focuses on developing robust models for natural language-guided cross-view image retrieval. This track challenges participants to create systems that can effectively retrieve corresponding images from large-scale cross-view databases based on natural language descriptions, even under common corruptions such as blurriness, occlusions, or sensory noise.

### 🎯 Task Description

Given natural language descriptions, participants need to:
- Retrieve the most relevant images from a large-scale image database
- Handle multi-platform imagery (drone, satellite, ground cameras)
- Maintain robustness under real-world corruptions and noise
- Achieve high precision in cross-modal matching

### 🎯 Objectives

- **Cross-View Retrieval**: Match images across drastically different viewpoints (aerial drone/satellite vs. ground-level)
- **Natural Language Guidance**: Process text descriptions to guide image retrieval
- **Robustness**: Maintain performance under real-world corruptions and noise
- **Multi-Platform Support**: Handle imagery from drone, satellite, and ground cameras

## Competition Details

- **Venue**: IROS 2025, Hangzhou (Oct 19-25, 2025)
- **Registration**: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdwfvk-NHdQh9-REiBLCjHMcyLT-sPCOCzJU-ux5jbcZLTkBg/viewform) (Open until Aug 15)
- **Contact**: robosense2025@gmail.com

### 🏆 **Awards**

| Prize | Award |
|:-|:-|
| 🥇 1st Place | $1000 + Certificate |
| 🥈 2nd Place | $600 + Certificate |
| 🥉 3rd Place | $400 + Certificate |
| 🌟 Innovation Award | Cash Award + Certificate |
| Participation | Certificate |

## 📊 Official Dataset

This track uses the **RoboSense Track 4 Cross-Modal Drone Navigation Dataset**, which is based on the GeoText-1652 benchmark and provides:

- **Multi-platform imagery**: drone, satellite, and ground cameras
- **Rich annotations**: global descriptions, bounding boxes, and spatial relations
- **Large scale**: 100K+ images across 72 universities
- **No overlap**: Training (33 universities) and test (39 universities) are completely separate

### Dataset Statistics

| Platform | Split | Images | Descriptions | Bbox-Texts | Classes | Universities |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| **Drone** | Train | 37,854 | 113,562 | 113,367 | 701 | 33 |
| **Drone** | Test | 51,355 | 154,065 | 140,179 | 951 | 39 |
| **Satellite** | Train | 701 | 2,103 | 1,709 | 701 | 33 |
| **Satellite** | Test | 951 | 2,853 | 2,006 | 951 | 39 |
| **Ground** | Train | 11,663 | 34,989 | 14,761 | 701 | 33 |
| **Ground** | Test | 2,921 | 8,763 | 4,023 | 793 | 39 |

### Baseline Performance (Phase 1 - 24GB GPU Version)
> **Note**: For Phase 1 evaluation, we recommend using the 24GB GPU version (~190 test cases) for faster development and testing.

| Text Query |  |  | Image Query |  |  |
|:-:|:-:|:-:|:-:|:-:|:-:|
| R@1 | R@5 | R@10 | R@1 | R@5 | R@10 |
| 29.9 | 46.3 | 54.1 | 50.1 | 81.2 | 90.3 |

## 🚀 Quick Start

### Prerequisites
- CUDA-capable GPU (24 GB+ recommended)
- Python 3.8+
- Git LFS

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/robosense2025/track4.git
   cd track4
   ```

2. **Set up environment**:
   ```bash
   conda create -n robosense_track4 python=3.8
   conda activate robosense_track4
   pip install -r requirements.txt
   ```

3. **Install Git LFS** (for large files):
   ```bash
   apt install git-lfs  # or brew install git-lfs on macOS
   git lfs install
   ```

4. **Download dataset and model**:
   ```bash
   # Official RoboSense Track 4 Dataset
   git clone https://huggingface.co/datasets/robosense/datasets
   
   # Pre-trained baseline model
   git clone https://huggingface.co/truemanv5666/GeoText1652_model
   ```

5. **Extract dataset** (if compressed):
   ```bash
   cd datasets/track4-cross-modal-drone-navigation
   # Extract any compressed files if present
   find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
   ```

6. **Configure paths**:
   - Update `re_bbox.yaml` with correct dataset paths
   - Update `method/configs/config_swinB_384.json` with checkpoint path

## 🔧 Usage

> **📁 Important**: All training, evaluation, and submission generation scripts are located in the `Method/` directory.

### Data Format

#### Input Data

**1. Test Queries File (`test_queries.txt`)**
```
q_00001	The image shows a view of a large, historic campus with several buildings and trees...
q_00002	The main object in the center of the image is a large building with a white roof...
q_00003	The image shows a college campus with a large building in the center...
```
- Tab-separated format
- Column 1: Query ID (q_00001, q_00002, ...)
- Column 2: Natural language description text

**2. Image Database (`Track_4_Phase_I_Images/`)**
```
Track_4_Phase_I_Images/
├── image_0a0bb9e5.jpeg
├── image_0a01df75.jpeg
├── image_7e591aa3.jpeg
└── ...
```
- Image naming: `image_` + hexadecimal hash + `.jpeg`
- Phase I contains ~190 test queries with corresponding images

#### Output Format

**Submission File (`submission.txt`)**
```
q1 image_7e591aa3 image_df1ee1e1 image_e58e691c image_7f384206 image_eaf35353 image_efb6995c image_2ec3e28a image_a6058312 image_f3223c63 image_b123683e
q2 image_45eccf3a image_61b3447d image_08eef12a image_24847f63 image_10f015dc image_ebfbc194 image_9c1361f3 image_00203915 image_7f8f1ebc image_8eff29a3
q3 image_f033c45d image_b670f69c image_c5f37733 image_ab6ea9a4 image_fe41c714 image_7c29de0b image_576e2a5f image_3652092d image_e15ce016 image_1358ff8f
```

**Format Requirements:**
- Each line format: `q{number} {image1} {image2} ... {image10}`
- Query ID mapping: `q_00001` → `q1`, `q_00002` → `q2`
- Image IDs: Only include `image_xxx` part, **without** `.jpeg` extension
- Ranking: Sorted by relevance (most relevant first)
- Count: Return **Top-10** most relevant images per query

### Evaluation (Recommended for Challenge Participants)

```bash
cd Method
python3 run.py --task "re_bbox" --dist "l4" --evaluate \
  --output_dir "output/eva" \
  --checkpoint "path/to/geotext_official_checkpoint.pth"
```

**Evaluation Options**:
- **Phase I (Recommended)** - 24GB GPU version (~190 cases): `datasets/track4-cross-modal-drone-navigation/test_24G_version.json`
- **Phase II** - Private test set (~190 cases): Available during Phase II evaluation only
- **Full test** (951 cases): `datasets/track4-cross-modal-drone-navigation/test_951_version.json`

### Generating Submission File

**All submission generation files are located in the `Method/` directory:**

```bash
cd Method
python generate_sub.py \
    --queries path/test_queries.txt \
    --checkpoint path/geotext_official_checkpoint.pth \
    --config config.yaml \
    --images path/Track_4_Phase_I_Images \
    --output submission.txt \
    --batch_size 32
```

**Parameters Explanation:**
- `--queries`: Test queries file (located in Method directory)
- `--checkpoint`: Pre-trained model checkpoint file
- `--config`: Configuration YAML file (in Method directory)
- `--images`: Path to test images dataset
- `--output`: Output submission file name
- `--batch_size`: Batch size for inference (adjust based on GPU memory)

### Training (For Model Development)

```bash
cd Method
nohup python3 run.py --task "re_bbox" --dist "l4" \
  --output_dir "output/train" \
  --checkpoint "path/to/geotext_official_checkpoint.pth" &
```

## 📁 Data Format

### JSON Annotation Example
```json
{
  "image_id": "0839/image-43.jpeg",
  "image": "train/0839/image-43.jpeg",
  "caption": "In the center of the image is a large, modern office building...",
  "sentences": [
    "The object in the center of the image is a large office building...",
    "On the upper middle side of the building, there is a street...",
    "On the middle right side of the building, there is a parking lot..."
  ],
  "bboxes": [
    [0.408688, 0.688366, 0.388595, 0.623482],
    [0.242049, 0.385560, 0.304881, 0.289198],
    [0.738844, 0.832005, 0.521311, 0.334470]
  ]
}
```

### Directory Structure
```
track4/
├── Method/                              # ⭐ ALL SCRIPTS ARE HERE
│   ├── generate_sub.py                  # Submission generation script
│   ├── run.py                          # Training/evaluation script
│   ├── config.yaml                     # Configuration file
│   ├── configs/                        # Model configurations
│   ├── models/                         # Model definitions
│   ├── dataset/                        # Data processing modules
│   └── utils/                          # Utility functions
├── Track_4_Phase_I_Images/             # Test images
│   ├── image_0a0bb9e5.jpeg
│   ├── image_0a01df75.jpeg
│   └── ...
├── datasets/track4-cross-modal-drone-navigation/
│   ├── train/
│   │   ├── 0001/
│   │   │   ├── drone_view.jpg
│   │   │   ├── street_view.jpg
│   │   │   └── satellite_view.jpg
│   │   └── .../
│   ├── test/
│   │   ├── gallery_no_train(250)/
│   │   └── query(701)/
│   ├── train.json
│   ├── test_951_version.json
│   └── test_queries.txt                # Test queries file
├── submission.txt                      # Generated submission file
└── requirements.txt
```

## 🎖️ Challenge Participation

### Submission Requirements
1. **Phase I**: Submit results on public test set with reproducible code
2. **Phase II**: Final evaluation on private test set (same size as Phase I)
3. **Code**: Submit reproducible code with your final results (all scripts in Method directory)
4. **Model**: Include trained model weights
5. **Report**: Technical report describing your approach

### Submission Generation Workflow
```bash
# Step 1: Navigate to Method directory
cd Method

# Step 2: Generate submission file
python generate_sub.py \
    --queries ../datasets/track4-cross-modal-drone-navigation/test_queries.txt \
    --checkpoint ../checkpoints/geotext_official_checkpoint.pth \
    --config config.yaml \
    --images ../Track_4_Phase_I_Images \
    --output submission.txt \
    --batch_size 32

# Step 3: Validate submission format
head -5 submission.txt
awk '{print NF}' submission.txt | sort | uniq -c  # Should show 11 fields per line
wc -l submission.txt  # Should show ~190 lines for Phase I

# Step 4: Package and submit
zip submission.zip submission.txt
# Upload submission.zip to competition platform
```

### Evaluation Metrics
- **Recall@K**: R@1, R@5, R@10 for both text-to-image and image-to-text retrieval
- **Robustness**: Performance under various corruptions and noise conditions
- **Phase 1**: Public leaderboard based on 24GB test set
- **Phase 2**: Final ranking based on private test set

### Timeline
- **Registration**: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdwfvk-NHdQh9-REiBLCjHMcyLT-sPCOCzJU-ux5jbcZLTkBg/viewform)
- **Phase 1 Deadline**: Public test set evaluation (~190 cases)
- **Phase 2 Deadline**: Private test set evaluation (~190 cases)
- **Awards Announcement**: IROS 2025

## 🔧 Model Development Guidelines

### Improvement Directions

1. **Feature Extractors**
   - Experiment with different visual encoders (ViT, CLIP, etc.)
   - Optimize text encoders (BERT, RoBERTa, etc.)

2. **Cross-Modal Fusion**
   - Attention mechanisms
   - Contrastive learning
   - Multi-level feature fusion

3. **Data Augmentation**
   - Image augmentation (rotation, scaling, color transformation)
   - Text augmentation (synonym replacement, back-translation)

4. **Robustness Enhancement**
   - Noise adaptation
   - Domain adaptation techniques
   - Multi-view consistency

### Validation Strategy
```bash
# Validate submission format
python -c "
import sys
with open('submission.txt', 'r') as f:
    lines = f.readlines()
    
# Check line count
print(f'Total queries: {len(lines)}')

# Check field count per line
field_counts = [len(line.strip().split()) for line in lines]
print(f'Fields per line: {set(field_counts)}')  # Should be {11}

# Check query ID format
query_ids = [line.split()[0] for line in lines]
print(f'Query IDs: {query_ids[:5]}...')  # Should be q1, q2, q3...

# Check image ID format
image_ids = []
for line in lines[:3]:
    image_ids.extend(line.split()[1:4])  # First 3 images from first 3 queries
print(f'Sample image IDs: {image_ids}')  # Should be image_xxx format
"
```

## ⚠️ Common Issues & Solutions

### Q1: Submission Format Error
**A**: Ensure each line contains exactly 11 fields, and image IDs don't include `.jpeg` extension

### Q2: Query ID Mapping Confusion
**A**: Map `q_00001` → `q1`, `q_00002` → `q2`, etc., in sequential order

### Q3: Image Path Issues
**A**: Verify `--images` parameter points to directory containing `image_*.jpeg` files

### Q4: Memory Issues
**A**: Reduce `--batch_size` parameter, start with 16 or 8 for limited GPU memory

### Q5: Performance Improvement
**A**: Focus on cross-modal feature alignment and similarity computation optimization

## 🔗 Resources

- **Challenge Website**: [robosense2025.github.io](https://robosense2025.github.io/)
- **Track 4 Details**: [Track 4 Page](https://robosense2025.github.io/track4)
- **Registration**: [Google Form](https://docs.google.com/forms/d/e/1FAIpQLSdwfvk-NHdQh9-REiBLCjHMcyLT-sPCOCzJU-ux5jbcZLTkBg/viewform)
- **Submission Platform**: [CodaBench](https://www.codabench.org/competitions/9219/)
- **Official Dataset**: [HuggingFace - RoboSense Track 4](https://huggingface.co/datasets/robosense/datasets/tree/main/track4-cross-modal-drone-navigation)
- **Original GeoText Paper**: [arXiv:2311.12751](https://arxiv.org/abs/2311.12751)
- **Baseline Model**: [HuggingFace](https://huggingface.co/truemanv5666/GeoText1652_model)

## 📧 Contact & Support

- **Email**: robosense2025@gmail.com
- **Official Website**: https://robosense2025.github.io
- **Issues**: Please use GitHub Issues for technical questions

## 📄 Citation

If you use the code and dataset in your research, please cite:

```bibtex
@inproceedings{chu2024towards, 
  title = {Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
  author = {Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
  booktitle = {European Conference on Computer Vision},
  year = {2024},
  organization = {Springer}
}
```

## 🎯 Submission Checklist

- [ ] Environment setup completed
- [ ] Data downloaded and paths configured correctly
- [ ] Successfully run `generate_sub.py`
- [ ] `submission.txt` format validation passed
- [ ] Correct number of queries (Phase I ~190)
- [ ] Correct image ID format (without extension)
- [ ] Compressed and uploaded to competition platform
- [ ] Awaiting evaluation results

## Acknowledgements

### RoboSense 2025 Challenge Organizers

<p align="center">
  <img src="docs/figures/organizers.jpg" align="center" width="99%">
</p>

### RoboSense 2025 Program Committee

<p align="center">
  <img src="docs/figures/organizers2.jpg" align="center" width="99%">
</p>

---

<div align="center">

**🤖 Ready to sense the world robustly? Register now and compete for $2,000!**

[**📝 Register Here**](https://docs.google.com/forms/d/e/1FAIpQLSdwfvk-NHdQh9-REiBLCjHMcyLT-sPCOCzJU-ux5jbcZLTkBg/viewform) | [**🌐 Challenge Website**](https://robosense2025.github.io/) | [**📧 Contact Us**](mailto:robosense2025.gmail.com)

Made with ❤️ by the RoboSense 2025 Team

</div>
