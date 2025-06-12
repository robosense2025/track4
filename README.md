# 🚁 RoboSense 2025 Challenge - Track 4: Cross-Modal Drone Navigation

<div align="center">

**Official Baseline Implementation for Cross-Modal Drone Navigation**

*Based on GeoText-1652: Towards Natural Language-Guided Drones with Spatial Relation Matching*

[![RoboSense Challenge](https://img.shields.io/badge/RoboSense-2025-blue)](https://robosense2025.github.io/)
[![Track 4](https://img.shields.io/badge/Track-Cross--Modal%20Drone%20Navigation-green)](https://robosense2025.github.io/track4)
[![IROS 2025](https://img.shields.io/badge/IROS-2025-red)](https://iros2025.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

**🏆 Prize Pool: $2,000 USD for Track 4 Winners**

</div>

## 📢 Challenge Overview

**Track 4: Cross-Modal Drone Navigation** focuses on developing robust models for natural language-guided cross-view image retrieval. This track challenges participants to create systems that can effectively retrieve corresponding images from large-scale cross-view databases based on natural language descriptions, even under common corruptions such as blurriness, occlusions, or sensory noise.

### 🎯 Challenge Objectives

- **Cross-View Retrieval**: Match images across drastically different viewpoints (aerial drone/satellite vs. ground-level)
- **Natural Language Guidance**: Process text descriptions to guide image retrieval
- **Robustness**: Maintain performance under real-world corruptions and noise
- **Multi-Platform Support**: Handle imagery from drone, satellite, and ground cameras

## 🏆 Competition Details

- **Venue**: IROS 2025, Hangzhou, China (Oct 19-25, 2025)
- **Registration**: [Google Form](https://forms.gle/robosense2025)
- **Contact**: robosense2025@gmail.com
- **Awards**: 
  - 🥇 1st Place: $1,000 + Certificate
  - 🥈 2nd Place: $600 + Certificate  
  - 🥉 3rd Place: $400 + Certificate
  - 🌟 Innovation Award: Certificate

## 📊 Official Dataset

This track uses the **RoboSense Track 4 Cross-Modal Drone Navigation Dataset**, which is based on the GeoText-1652 benchmark and provides:

- **Multi-platform imagery**: drone, satellite, and ground cameras
- **Rich annotations**: global descriptions, bounding boxes, and spatial relations
- **Large scale**: 100K+ images across 72 universities
- **No overlap**: Training (33 universities) and test (39 universities) are completely separate

### Dataset Statistics

| Platform | Split | Images | Descriptions | Bbox-Texts | Classes | Universities |
|----------|-------|--------|--------------|------------|---------|--------------|
| **Drone** | Train | 37,854 | 113,562 | 113,367 | 701 | 33 |
| **Drone** | Test | 51,355 | 154,065 | 140,179 | 951 | 39 |
| **Satellite** | Train | 701 | 2,103 | 1,709 | 701 | 33 |
| **Satellite** | Test | 951 | 2,853 | 2,006 | 951 | 39 |
| **Ground** | Train | 11,663 | 34,989 | 14,761 | 701 | 33 |
| **Ground** | Test | 2,921 | 8,763 | 4,023 | 793 | 39 |

### Baseline Performance (24GB GPU Version)
```
| Text Query | Image Query |
|R@1  R@5  R@10|R@1  R@5  R@10|
|29.9|46.3|54.1|50.1|81.2|90.3|
```

## 🚀 Quick Start

### Prerequisites
- CUDA-capable GPU (24GB+ recommended)
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

5. **Extract dataset images**:
   ```bash
   cd datasets/track4-cross-modal-drone-navigation/images
   find . -type f -name "*.tar.gz" -print0 | xargs -0 -I {} bash -c 'tar -xzf "{}" -C "$(dirname "{}")" && rm "{}"'
   ```

6. **Configure paths**:
   - Update `re_bbox.yaml` with correct dataset paths
   - Update `method/configs/config_swinB_384.json` with checkpoint path

## 🔧 Usage

### Evaluation (Recommended for Challenge Participants)

```bash
cd Method
python3 run.py --task "re_bbox" --dist "l4" --evaluate \
  --output_dir "output/eva" \
  --checkpoint "path/to/geotext_official_checkpoint.pth"
```

**Evaluation Options**:
- **Full test** (951 cases): `datasets/track4-cross-modal-drone-navigation/test_951_version.json`
- **24GB GPU version** (~190 cases): `datasets/track4-cross-modal-drone-navigation/test_24G_version.json`

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
datasets/track4-cross-modal-drone-navigation/
├── train/
│   ├── 0001/
│   │   ├── drone_view.jpg
│   │   ├── street_view.jpg
│   │   └── satellite_view.jpg
│   └── .../
├── test/
│   ├── gallery_no_train(250)/
│   └── query(701)/
├── train.json
└── test_951_version.json
```

## 🎖️ Challenge Participation

### Submission Requirements
1. **Code**: Submit reproducible code with your final results
2. **Model**: Include trained model weights
3. **Results**: Provide evaluation metrics on the official test set
4. **Report**: Technical report describing your approach

### Evaluation Metrics
- **Recall@K**: R@1, R@5, R@10 for both text-to-image and image-to-text retrieval
- **Robustness**: Performance under various corruptions and noise conditions

### Timeline
- **Registration**: [Google Form](https://forms.gle/robosense2025)
- **Phase 1 Deadline**: TBA
- **Phase 2 Deadline**: TBA  
- **Awards Announcement**: IROS 2025

## 🔗 Resources

- **Challenge Website**: [robosense2025.github.io](https://robosense2025.github.io/)
- **Track 4 Details**: [Track 4 Page](https://robosense2025.github.io/track4)
- **Official Dataset**: [HuggingFace - RoboSense Track 4](https://huggingface.co/datasets/robosense/datasets/tree/main/track4-cross-modal-drone-navigation)
- **Original GeoText Paper**: [arXiv:2311.12751](https://arxiv.org/pdf/2311.12751)
- **Baseline Model**: [HuggingFace](https://huggingface.co/truemanv5666/GeoText1652_model)

## 📧 Contact & Support

- **Email**: robosense2025@gmail.com
- **Challenge Website**: https://robosense2025.github.io/
- **Issues**: Please use GitHub Issues for technical questions

## 📄 Citation

If you use this baseline or the GeoText-1652 dataset, please cite:

```bibtex
@inproceedings{chu2024towards, 
  title={Towards Natural Language-Guided Drones: GeoText-1652 Benchmark with Spatial Relation Matching}, 
  author={Chu, Meng and Zheng, Zhedong and Ji, Wei and Wang, Tingyu and Chua, Tat-Seng}, 
  booktitle={ECCV}, 
  year={2024} 
}
```

## 🙏 Acknowledgements

- **GeoText-1652 Team** for the original benchmark and baseline implementation
- **X-VLM** project for the foundational vision-language model
- **RoboSense Challenge Organizers** for hosting this competition

---

<div align="center">

**🚁 Ready to navigate the future of drone intelligence? Register now and compete for $2,000!**

[**📝 Register Here**](https://forms.gle/robosense2025) | [**🌐 Challenge Website**](https://robosense2025.github.io/) | [**📧 Contact Us**](mailto:robosense2025@gmail.com)

Made with ❤️ by the RoboSense 2025 Team

</div>
